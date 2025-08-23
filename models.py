"""
Django models for OpenAPI specification management.

This module provides models and utilities for importing, storing, and exporting
OpenAPI specifications in a Django application.
"""

import copy
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import yaml
from django.db import models
from django.db.models import Count


# Constants for OpenAPI required fields
REQUIRED_KEYS = {
    "root": ["openapi", "info", "paths"],
    "info": ["title", "version"],
    "operation": ["responses"],
    "parameter": ["name", "in"],
    "response": ["description"],
}

# Valid HTTP method contexts for OpenAPI operations
HTTP_METHOD_CONTEXTS = (
    "get",
    "post",
    "put",
    "delete",
    "patch",
    "options",
    "head",
    "trace",
)

# Default stable keys for sorting normalized objects
DEFAULT_STABLE_KEYS = ["name", "status_code", "in"]


@dataclass
class SeenObjects:
    """Container for tracking seen objects during import."""

    endpoints: List[int]
    parameters: List[int]
    responses: List[int]


def strip_optional(obj: Any, context: str = "root") -> Any:
    """
    Recursively strip out optional fields, keeping only those required by OpenAPI spec.

    Args:
        obj: The object to strip (dict, list, or primitive)
        context: Current context in the OpenAPI structure

    Returns:
        Object with only required fields retained
    """
    if not isinstance(obj, (dict, list)):
        return obj

    if isinstance(obj, list):
        return [strip_optional(x, context) for x in obj]

    # Handle dict cases with a mapping approach to reduce return statements
    strip_handlers = {
        "root": lambda: {
            k: strip_optional(v, k)
            for k, v in obj.items()
            if k in REQUIRED_KEYS["root"]
        },
        "info": lambda: {k: v for k, v in obj.items() if k in REQUIRED_KEYS["info"]},
        "parameters": lambda: [strip_optional(p, "parameter") for p in obj],
        "responses": lambda: {
            code: strip_optional(r, "response")
            for code, r in normalize_responses(obj).items()
        },
        "response": lambda: {
            k: v for k, v in obj.items() if k in REQUIRED_KEYS["response"]
        },
        "parameter": lambda: {
            k: v for k, v in obj.items() if k in REQUIRED_KEYS["parameter"]
        },
    }

    # Check if context is an HTTP method
    if context in HTTP_METHOD_CONTEXTS:
        return {
            k: strip_optional(v, k)
            for k, v in obj.items()
            if k in REQUIRED_KEYS["operation"]
        }

    # Use handler if available, otherwise recurse
    handler = strip_handlers.get(context)
    if handler:
        return handler()

    # Default: keep recursing
    return {k: strip_optional(v, k) for k, v in obj.items()}


def normalize(obj: Any, stable_keys: Optional[List[str]] = None) -> Any:
    """
    Recursively normalize dicts/lists for stable comparison.

    Args:
        obj: Object to normalize
        stable_keys: Keys to use for stable sorting of dict lists

    Returns:
        Normalized object with consistent ordering
    """
    if stable_keys is None:
        stable_keys = DEFAULT_STABLE_KEYS

    if isinstance(obj, dict):
        return {k: normalize(v, stable_keys) for k, v in sorted(obj.items())}

    if isinstance(obj, list):
        if all(isinstance(x, dict) for x in obj):
            normalized_items = [normalize(x, stable_keys) for x in obj]

            # Prefer stable keys when available
            for stable_key in stable_keys:
                if all(
                    isinstance(item, dict) and stable_key in item
                    for item in normalized_items
                ):
                    return sorted(
                        normalized_items, key=lambda d: str(d.get(stable_key, ""))
                    )

            # Fallback: sort by JSON string
            return sorted(normalized_items, key=lambda d: json.dumps(d, sort_keys=True))

        normalized_items = [normalize(x, stable_keys) for x in obj]
        try:
            return sorted(normalized_items)
        except TypeError:
            # If items aren't sortable, return as-is
            return normalized_items

    return obj


def normalize_responses(responses: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure responses always have at least one entry.

    OpenAPI requires at least one response definition.

    Args:
        responses: Dictionary of response definitions

    Returns:
        Normalized responses with at least a default entry
    """
    if not responses:
        return {"default": {"description": ""}}
    return responses


class API(models.Model):
    """Represents an API service with OpenAPI specification support."""

    name = models.CharField(max_length=128, help_text="Human-readable name of the API")
    version = models.CharField(
        max_length=32, blank=True, help_text="Semantic or internal version string"
    )
    base_url = models.URLField(
        help_text="Base URL for all endpoints, e.g. https://api.example.com/v1"
    )

    class Meta:
        verbose_name = "API"
        verbose_name_plural = "APIs"
        ordering = ["name", "version"]

    def __str__(self) -> str:
        return f"{self.name} v{self.version}"

    def to_openapi(self) -> Dict[str, Any]:
        """
        Export API into OpenAPI 3.1 dict.

        Returns:
            OpenAPI specification as a dictionary
        """
        paths = {}
        for endpoint in self.endpoints.all():
            if endpoint.path not in paths:
                paths[endpoint.path] = {}
            paths[endpoint.path].update(endpoint.to_openapi())

        return {
            "openapi": "3.1.0",
            "info": {
                "title": self.name,
                "version": self.version or "1.0.0",
            },
            "servers": [{"url": self.base_url}] if self.base_url else [],
            "paths": paths,
        }

    def to_openapi_yaml(self) -> str:
        """
        Export API spec to YAML string.

        Returns:
            YAML representation of the OpenAPI specification
        """
        return yaml.dump(self.to_openapi(), sort_keys=False, default_flow_style=False)

    @classmethod
    def from_openapi_json(cls, json_file: str) -> "API":
        """
        Import API from an OpenAPI JSON file.

        Args:
            json_file: Path to the JSON file

        Returns:
            Created or updated API instance
        """
        with open(json_file, "r", encoding="utf-8") as file:
            spec = json.load(file)

        # Create or update API
        api, _ = cls.objects.update_or_create(
            name=spec["info"]["title"],
            defaults={
                "version": spec["info"].get("version", ""),
                "base_url": spec.get("servers", [{}])[0].get("url", ""),
            },
        )

        # Track seen objects for cleanup
        seen = SeenObjects(endpoints=[], parameters=[], responses=[])

        # Process paths and methods
        for path, methods in spec.get("paths", {}).items():
            for method, details in methods.items():
                if method.startswith("x-"):  # Skip vendor extensions
                    continue

                endpoint_data = {"path": path, "method": method, "details": details}
                cls._create_endpoint(api, endpoint_data, seen)

        # Cleanup stale objects
        cls._cleanup_stale_objects(api, seen)

        return api

    @classmethod
    def _create_endpoint(
        cls,
        api: "API",
        endpoint_data: Dict[str, Any],
        seen: SeenObjects,
    ) -> None:
        """Create or update an endpoint with its parameters and responses.

        Args:
            api: The API instance
            endpoint_data: Dict containing 'path', 'method', and 'details'
            seen: Container for tracking seen objects
        """
        path = endpoint_data["path"]
        method = endpoint_data["method"]
        details = endpoint_data["details"]

        http_method, _ = HttpMethod.objects.get_or_create(name=method.upper())

        endpoint, _ = Endpoint.objects.update_or_create(
            api=api,
            path=path,
            method=http_method,
            defaults={
                "summary": details.get("summary", ""),
                "description": details.get("description", ""),
            },
        )
        seen.endpoints.append(endpoint.id)

        # Process parameters
        cls._process_parameters(
            endpoint, details.get("parameters", []), seen.parameters
        )

        # Process request body
        if "requestBody" in details:
            cls._process_request_body(endpoint, details["requestBody"], seen.parameters)

        # Process responses
        cls._process_responses(endpoint, details.get("responses", {}), seen.responses)

    @classmethod
    def _process_parameters(
        cls,
        endpoint: "Endpoint",
        parameters: List[Dict[str, Any]],
        seen_parameters: List[int],
    ) -> None:
        for param_spec in parameters:
            scope, _ = ParameterScope.objects.get_or_create(name=param_spec["in"])
            schema = param_spec.get("schema", {})
            obj_dict = {
                "required": param_spec.get("required", False),
                "data_type": schema.get("type", ""),
                "data_format": schema.get("format", ""),
                "description": param_spec.get("description", ""),
            }

            try:
                param, created = Parameter.objects.get_or_create(
                    endpoint=endpoint,
                    name=param_spec["name"],
                    scope=scope,
                    defaults=obj_dict,
                )

                if not created:
                    for key, value in obj_dict.items():
                        setattr(param, key, value)
                    param.save()

                seen_parameters.append(param.id)
            except:
                print(f"Failed to create parameter {param_spec['name']}")
                print(param_spec)


    @classmethod
    def _process_request_body(
        cls,
        endpoint: "Endpoint",
        request_body: Dict[str, Any],
        seen_parameters: List[int],
    ) -> None:
        """Process request body as a special parameter."""
        scope, _ = ParameterScope.objects.get_or_create(name="body")
        content = request_body.get("content", {})
        schema = {}

        if "application/json" in content:
            schema = content["application/json"].get("schema", {})

        obj_dict = {
            "required": request_body.get("required", False),
            "data_type": schema.get("type", ""),
            "data_format": schema.get("format", ""),
            "description": request_body.get("description", ""),
        }
        # Use scope_id to avoid ForeignKey issues in get_or_create
        body_param, created = Parameter.objects.get_or_create(
            endpoint=endpoint, name="body", scope=scope, defaults=obj_dict
        )

        # If not created, update the existing parameter
        if not created:
            for key, value in obj_dict.items():
                setattr(body_param, key, value)
            body_param.save()

        seen_parameters.append(body_param.id)

    @classmethod
    def _process_responses(
        cls, endpoint: "Endpoint", responses: Dict[str, Any], seen_responses: List[int]
    ) -> None:
        """Process and create response objects."""
        for status_code, resp_spec in responses.items():
            schema = {}
            example = ""

            content = resp_spec.get("content", {})
            if "application/json" in content:
                json_content = content["application/json"]
                schema = json_content.get("schema", {})

                if "example" in json_content:
                    example = json.dumps(json_content["example"], indent=2)

            response, created = Response.objects.get_or_create(
                endpoint=endpoint,
                status_code=status_code,
                defaults={
                    "description": resp_spec.get("description", ""),
                    "data_type": schema.get("type", ""),
                    "data_format": schema.get("format", ""),
                    "example": example,
                },
            )

            # If not created, update the existing response
            if not created:
                response.description = resp_spec.get("description", "")
                response.data_type = schema.get("type", "")
                response.data_format = schema.get("format", "")
                response.example = example
                response.save()

            seen_responses.append(response.id)

    @classmethod
    def _cleanup_stale_objects(
        cls,
        api: "API",
        seen: SeenObjects,
    ) -> None:
        """Remove objects that weren't seen during import."""
        api.endpoints.exclude(id__in=seen.endpoints).delete()
        Parameter.objects.filter(endpoint__api=api).exclude(
            id__in=seen.parameters
        ).delete()
        Response.objects.filter(endpoint__api=api).exclude(
            id__in=seen.responses
        ).delete()

    @classmethod
    def checksum(
        cls, json_file: str, verbose: bool = False
    ) -> Union[bool, Dict[str, Any]]:
        """
        Verify that import/export cycle preserves essential OpenAPI data.

        Args:
            json_file: Path to the JSON file to check
            verbose: If True, return detailed failure information

        Returns:
            True if checksum passes, False or dict of failures otherwise
        """
        with open(json_file, "r", encoding="utf-8") as file:
            original = json.load(file)

        api = cls.from_openapi_json(json_file)
        exported = api.to_openapi()

        norm_orig = normalize(strip_optional(copy.deepcopy(original)))
        norm_export = normalize(strip_optional(exported))

        if norm_orig == norm_export:
            return True

        if not verbose:
            return False

        # Collect failing endpoints for verbose output
        failed = {}
        orig_paths = norm_orig.get("paths", {})
        export_paths = norm_export.get("paths", {})

        for path, methods in orig_paths.items():
            exp_methods = export_paths.get(path, {})
            if normalize(methods) != normalize(exp_methods):
                failed[path] = {"original": methods, "exported": exp_methods}

        return failed if failed else False

    def get_stats(self) -> Dict[str, Any]:
        """
        Get API statistics.

        Returns:
            Dict with 'path_count', 'methods' (count per HTTP method),
            and 'path_method_distribution' (paths per method count)
        """

        path_count = self.endpoints.values("path").distinct().count()

        method_counts = dict(
            self.endpoints.values_list("method__name")
            .annotate(count=Count("method__name"))
            .order_by("method__name")
        )

        # Get distribution of paths by number of methods
        path_method_counts = (
            self.endpoints.values("path")
            .annotate(method_count=Count("method"))
            .values_list("method_count", flat=True)
        )

        distribution = {}
        for count in path_method_counts:
            distribution[count] = distribution.get(count, 0) + 1

        return {
            "path_count": path_count,
            "methods": method_counts,
            "path_method_distribution": dict(sorted(distribution.items())),
        }


class HttpMethod(models.Model):
    """HTTP methods (GET, POST, etc.) as database records."""

    name = models.CharField(max_length=16, unique=True)
    description = models.TextField(blank=True)

    class Meta:
        verbose_name = "HTTP Method"
        verbose_name_plural = "HTTP Methods"
        ordering = ["name"]

    def __str__(self) -> str:
        return self.name


class ParameterScope(models.Model):
    """Defines where a parameter belongs (query, header, path, body, etc.)."""

    name = models.CharField(max_length=32, unique=True)
    description = models.TextField(blank=True)

    class Meta:
        verbose_name = "Parameter Scope"
        verbose_name_plural = "Parameter Scopes"
        ordering = ["name"]

    def __str__(self) -> str:
        return self.name


class Endpoint(models.Model):
    """Represents an API endpoint with a specific path and HTTP method."""

    api = models.ForeignKey(API, on_delete=models.CASCADE, related_name="endpoints")
    path = models.CharField(max_length=255)
    method = models.ForeignKey(HttpMethod, on_delete=models.PROTECT)
    summary = models.CharField(max_length=255, blank=True)
    description = models.TextField(blank=True)

    class Meta:
        verbose_name = "Endpoint"
        verbose_name_plural = "Endpoints"
        unique_together = ("api", "path", "method")
        ordering = ["path", "method"]

    def __str__(self) -> str:
        return f"{self.method} {self.path}"

    def to_openapi(self) -> Dict[str, Any]:
        """
        Export this endpoint to OpenAPI dict format.

        Returns:
            Dictionary with method as key and OpenAPI operation as value
        """
        parameters = []
        request_body = None

        # Process parameters
        for param in self.parameters.all():
            if param.scope.name == "body":
                request_body = self._build_request_body(param)
            else:
                parameters.append(self._build_parameter(param))

        # Process responses
        responses = self._build_responses()

        # Build operation object
        operation = {
            "summary": self.summary or "",
            "description": self.description or "",
            "parameters": parameters,
            "responses": responses or {"default": {"description": ""}},
        }

        if request_body:
            operation["requestBody"] = request_body

        return {self.method.name.lower(): operation}

    @staticmethod
    def _build_parameter(param: "Parameter") -> Dict[str, Any]:
        """Build OpenAPI parameter object."""
        param_obj = {
            "name": param.name,
            "in": param.scope.name,
            "required": param.required,
            "description": param.description or "",
            "schema": {"type": param.data_type or "string"},
        }

        if param.data_format:
            param_obj["schema"]["format"] = param.data_format

        return param_obj

    @staticmethod
    def _build_request_body(param: "Parameter") -> Dict[str, Any]:
        """Build OpenAPI request body object."""
        request_body = {
            "description": param.description or "",
            "required": param.required,
            "content": {
                "application/json": {"schema": {"type": param.data_type or "string"}}
            },
        }

        if param.data_format:
            request_body["content"]["application/json"]["schema"][
                "format"
            ] = param.data_format

        return request_body

    def _build_responses(self) -> Dict[str, Any]:
        """Build OpenAPI responses object."""
        responses = {}

        for response in self.responses.all():
            response_obj = {
                "description": response.description or "",
                "content": {
                    "application/json": {
                        "schema": {"type": response.data_type or "string"}
                    }
                },
            }

            if response.data_format:
                response_obj["content"]["application/json"]["schema"][
                    "format"
                ] = response.data_format

            if response.example:
                try:
                    response_obj["content"]["application/json"]["example"] = (
                        yaml.safe_load(response.example)
                    )
                except (yaml.YAMLError, ValueError):
                    # If example isn't valid YAML/JSON, skip it
                    pass

            responses[response.status_code] = response_obj

        return responses


class Parameter(models.Model):
    """Represents a parameter for an endpoint, including request body."""

    endpoint = models.ForeignKey(
        Endpoint, on_delete=models.CASCADE, related_name="parameters"
    )
    name = models.CharField(max_length=64)
    scope = models.ForeignKey(ParameterScope, on_delete=models.PROTECT)
    required = models.BooleanField(default=False)
    data_type = models.CharField(
        max_length=32, blank=True, help_text="OpenAPI data type (string, integer, etc.)"
    )
    data_format = models.CharField(
        max_length=32, blank=True, help_text="OpenAPI format (date-time, email, etc.)"
    )
    description = models.TextField(blank=True)

    class Meta:
        verbose_name = "Parameter"
        verbose_name_plural = "Parameters"
        unique_together = ("endpoint", "name", "scope")
        ordering = ["name"]

    def __str__(self) -> str:
        return f"{self.name} ({self.scope})"


class Response(models.Model):
    """Represents a response for an endpoint."""

    endpoint = models.ForeignKey(
        Endpoint, on_delete=models.CASCADE, related_name="responses"
    )
    status_code = models.CharField(
        max_length=5, help_text="HTTP status code or 'default'"
    )
    description = models.TextField(blank=True)
    data_type = models.CharField(
        max_length=32, blank=True, help_text="Response data type"
    )
    data_format = models.CharField(
        max_length=32, blank=True, help_text="Response data format"
    )
    example = models.TextField(blank=True, help_text="Example response (JSON format)")

    class Meta:
        verbose_name = "Response"
        verbose_name_plural = "Responses"
        unique_together = ("endpoint", "status_code")
        ordering = ["status_code"]

    def __str__(self) -> str:
        return f"{self.status_code} for {self.endpoint}"
