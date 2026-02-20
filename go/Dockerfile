# Build stage
FROM golang:1.25-bookworm AS builder

WORKDIR /app

# Copy go.mod and go.sum first for caching
COPY go.mod go.sum ./
RUN go mod download

# Copy the rest of the code
COPY . .

# Build the server binary
RUN CGO_ENABLED=0 GOOS=linux go build -o server ./cmd/server/main.go

# Final stage
FROM debian:bookworm-slim

WORKDIR /app

# Install CA certificates for external downloads if needed
RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*

# Copy the binary from builder
COPY --from=builder /app/server .

# Copy the models directory for weights
COPY ./models ./models

# Set environment variables
ENV PORT=7860
ENV MODEL_PATH=models/latest_checkpoint.json

# Expose the port
EXPOSE 7860

# Run the server
CMD ["./server"]
