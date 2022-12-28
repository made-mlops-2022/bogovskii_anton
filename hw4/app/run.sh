#!/bin/bash

{
  sleep 90
  kill $$
} &

sleep 30 && uvicorn inference_service:app --host 0.0.0.0
