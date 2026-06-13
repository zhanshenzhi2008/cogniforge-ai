"""
OpenTelemetry Tracing Configuration
"""
import os
import logging
import colorlog
from contextvars import ContextVar
from typing import Optional
from datetime import datetime

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.resources import Resource, SERVICE_NAME
from opentelemetry.trace import Span, Status, StatusCode
from opentelemetry.propagate import set_global_textmap
from opentelemetry.propagators.b3 import B3MultiFormat

logger = logging.getLogger(__name__)

# Context variable to store current trace_id for easy access
current_trace_id: ContextVar[Optional[str]] = ContextVar("current_trace_id", default=None)


class TraceIdLogFilter(logging.Filter):
    """
    Log filter that adds trace_id to each log record.
    """
    
    def filter(self, record: logging.LogRecord) -> bool:
        # Get trace_id from context
        trace_id = current_trace_id.get()
        if trace_id:
            record.trace_id = trace_id
        else:
            record.trace_id = "no-trace"
        return True


class TraceIdLogFormatter(colorlog.ColoredFormatter):
    """
    Colored log formatter that includes trace_id in the log output.
    """
    
    def __init__(self, *args, **kwargs):
        # Default colors for log levels
        kwargs.setdefault('log_colors', {
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        })
        super().__init__(*args, **kwargs)
    
    def format(self, record: logging.LogRecord) -> str:
        # Get trace_id from record (set by TraceIdLogFilter)
        trace_id = getattr(record, 'trace_id', 'no-trace')
        record.tracemsg = f"[{trace_id[:16]}] {record.getMessage()}"
        return super().format(record)


def setup_trace_logging():
    """
    Configure logging to include trace_id in all log messages with color support.
    Call this after init_tracing() to enable trace-aware logging.
    """
    # Get root logger
    root_logger = logging.getLogger()
    
    # Add our filter to root logger (avoid duplicate filters)
    if not any(isinstance(f, TraceIdLogFilter) for f in root_logger.filters):
        root_logger.addFilter(TraceIdLogFilter())
    
    # Update existing handlers with colored trace-aware formatter
    for handler in root_logger.handlers:
        # Use TraceIdLogFormatter with colors (colors set in class __init__)
        formatter = TraceIdLogFormatter(
            fmt='%(asctime)s [%(levelname)s] %(name)s [%(tracemsg)s]',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)


def init_tracing(service_name: str = "cogniforge-ai", app=None) -> None:
    """
    Initialize OpenTelemetry tracing.
    
    Supports:
    - Console export (OTEL_SDK_EXPORT=console)
    - OTLP export (default, for collector)
    - Jaeger export (OTEL_EXPORTER=jaeger)
    
    Args:
        service_name: Name of the service for tracing
        app: Optional FastAPI app instance for instrumentation. If None, 
             instrumentation should be done separately after app creation.
    """
    # Create resource with service info
    resource = Resource.create({
        SERVICE_NAME: service_name,
        "service.version": "1.0.0",
        "deployment.environment": os.getenv("ENVIRONMENT", "development"),
    })
    
    # Create tracer provider
    provider = TracerProvider(resource=resource)
    
    # Configure exporter based on environment
    exporter_type = os.getenv("OTEL_EXPORTER", "console").lower()
    
    if exporter_type == "console":
        # Console exporter for development
        processor = BatchSpanProcessor(ConsoleSpanExporter())
        provider.add_span_processor(processor)
        logger.info("OpenTelemetry console exporter configured")
        
    elif exporter_type == "otlp":
        # OTLP exporter for production (requires otel-collector)
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
            
            otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
            exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
            processor = BatchSpanProcessor(exporter)
            provider.add_span_processor(processor)
            logger.info(f"OpenTelemetry OTLP exporter configured: {otlp_endpoint}")
        except ImportError:
            logger.warning("OTLP exporter not available, using console exporter")
            processor = BatchSpanProcessor(ConsoleSpanExporter())
            provider.add_span_processor(processor)
            
    elif exporter_type == "jaeger":
        # Jaeger exporter
        try:
            from opentelemetry.exporter.jaeger.thrift import JaegerExporter
            
            jaeger_endpoint = os.getenv("JAEGER_AGENT_HOST", "localhost")
            jaeger_port = int(os.getenv("JAEGER_AGENT_PORT", "6831"))
            exporter = JaegerExporter(
                agent_host_name=jaeger_endpoint,
                agent_port=jaeger_port,
            )
            processor = BatchSpanProcessor(exporter)
            provider.add_span_processor(processor)
            logger.info(f"OpenTelemetry Jaeger exporter configured: {jaeger_endpoint}:{jaeger_port}")
        except ImportError:
            logger.warning("Jaeger exporter not available, install opentelemetry-exporter-jaeger")
            processor = BatchSpanProcessor(ConsoleSpanExporter())
            provider.add_span_processor(processor)
    
    else:
        # Default to console
        processor = BatchSpanProcessor(ConsoleSpanExporter())
        provider.add_span_processor(processor)
    
    # Set global tracer provider
    trace.set_tracer_provider(provider)
    
    # Configure B3 propagator to support X-Trace-ID header
    set_global_textmap(B3MultiFormat())
    logger.info("B3 propagator configured for X-Trace-ID support")
    
    # Instrument FastAPI (only if app is provided)
    if app is not None:
        try:
            from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
            
            def add_trace_id_header(request, request_span: Span):
                """Add trace_id to response headers after each request."""
                trace_id = format_trace_id(request_span)
                if trace_id:
                    current_trace_id.set(trace_id)
            
            FastAPIInstrumentor.instrument_app(
                app=app,
                excluded_urls="health,ready,live",
            )
            logger.info("FastAPI instrumentation enabled")
        except ImportError:
            logger.warning("FastAPI instrumentation not available")
    else:
        logger.info("FastAPI app not provided, skipping instrumentation (call instrument_app separately)")
    
    # Instrument httpx for outbound HTTP calls
    try:
        from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
        HTTPXClientInstrumentor().instrument()
        logger.info("HTTPX instrumentation enabled")
    except ImportError:
        logger.warning("HTTPX instrumentation not available")


def get_tracer(name: str = __name__) -> trace.Tracer:
    """Get a tracer instance."""
    return trace.get_tracer(name)


def get_current_trace_id() -> Optional[str]:
    """
    Get the current trace_id from context.
    
    Returns:
        The trace_id as a hex string, or None if not in a trace.
    """
    # First try ContextVar
    trace_id = current_trace_id.get()
    if trace_id:
        return trace_id
    
    # Fall back to current span
    span = trace.get_current_span()
    if span and span.get_span_context().is_valid:
        return format_trace_id(span)
    
    return None


def format_trace_id(span_or_context) -> Optional[str]:
    """Extract and format trace_id from span or span context."""
    if hasattr(span_or_context, "get_span_context"):
        ctx = span_or_context.get_span_context()
    else:
        ctx = span_or_context
    
    if ctx and ctx.is_valid:
        return format(ctx.trace_id, "032x")
    return None


def create_span(
    name: str,
    attributes: Optional[dict] = None,
    kind: trace.SpanKind = trace.SpanKind.INTERNAL,
):
    """
    Create a new span as child of current span.
    
    Usage:
        with create_span("my_operation", {"key": "value"}) as span:
            # do work
            span.set_attribute("result", "success")
    """
    tracer = get_tracer()
    return tracer.start_as_current_span(name, kind=kind, attributes=attributes)


def add_span_attributes(**attributes):
    """Add attributes to the current span."""
    span = trace.get_current_span()
    if span:
        for key, value in attributes.items():
            span.set_attribute(key, value)


def record_exception(exception: Exception, attributes: Optional[dict] = None):
    """Record an exception on the current span."""
    span = trace.get_current_span()
    if span:
        span.record_exception(exception, attributes=attributes)
        span.set_status(Status(StatusCode.ERROR, str(exception)))


class TracingContextMiddleware:
    """
    Middleware to extract trace context and make trace_id available.
    
    FastAPI/Starlette middleware that:
    1. Extracts trace context from incoming headers (B3 format)
    2. Makes trace_id available via ContextVar
    3. Adds trace_id to response headers
    """
    
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        # Extract trace_id from headers (B3 format uses X-B3-TraceId)
        headers = dict(scope.get("headers", []))
        
        trace_id = None
        for key, value in headers.items():
            if key in (b"x-b3-traceid", b"x-trace-id", b"x-request-id"):
                trace_id = value.decode("utf-8")
                break
        
        # Store in ContextVar
        if trace_id:
            current_trace_id.set(trace_id)
        
        # Response headers to capture trace_id
        response_headers = []
        
        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                # Add trace_id to response headers
                nonlocal response_headers
                if trace_id:
                    response_headers.append((b"x-trace-id", trace_id.encode("utf-8")))
                message["headers"] = list(message.get("headers", [])) + response_headers
            await send(message)
        
        await self.app(scope, receive, send_wrapper)


def get_trace_id_header_name() -> str:
    """
    Get the header name used for trace propagation.
    
    Returns:
        Header name for trace_id (统一使用 X-Trace-ID).
    """
    return "X-Trace-ID"


def inject_trace_context(carrier: dict) -> dict:
    """
    Inject trace context into a carrier dict (e.g., for HTTP headers or message metadata).
    
    Usage:
        headers = {}
        inject_trace_context(headers)
        # headers now contains trace propagation headers
    """
    from opentelemetry.propagate import inject
    inject(carrier)
    return carrier
