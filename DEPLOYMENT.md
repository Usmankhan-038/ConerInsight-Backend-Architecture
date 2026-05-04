# Deployment Notes

This backend uses TensorFlow, scikit-learn, OpenCV, ONNX Runtime, and model files. That stack is too large for Vercel Python serverless functions, which have a 500 MB ephemeral storage limit.

## Recommended setup

Deploy the backend on a server/container host such as Render, Railway, Fly.io, or a VPS, then keep your frontend on Vercel.

## Docker deploy

Use the included `Dockerfile`.

Start locally:

```bash
docker build -t keratoconus-backend .
docker run -p 8000:8000 --env-file .env keratoconus-backend
```

Health check:

```bash
curl http://localhost:8000/health
```

## Why Vercel fails

Vercel is not failing because of the two local model files. They are only a few MB. The large bundle comes from Python packages, mainly TensorFlow and related ML/runtime dependencies.

The `.vercelignore` file removes generated files and local output from the deployment upload, but it cannot make TensorFlow fit inside Vercel's Python function storage limit.
