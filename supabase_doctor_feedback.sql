alter table public.predictions
add column if not exists doctor_feedback text;
