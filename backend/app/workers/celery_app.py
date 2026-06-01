from celery import Celery
from app.config import get_settings

settings = get_settings()

celery_app = Celery(
    "videosync",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=["app.workers.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    # Prevent one stuck task from blocking the worker
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    # Retry failed tasks up to 3 times with exponential backoff
    task_max_retries=3,
    task_default_retry_delay=10,
    # Expire results after 1 hour to prevent Redis memory bloat
    result_expires=3600,
    # Soft time limit: warn at None min; hard kill at 30 min
    # (SeSyn-Net on CPU can take a few minutes for long videos)
    task_soft_time_limit=None,
    task_time_limit=1800,
)
