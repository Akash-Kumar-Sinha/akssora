# Build stage
FROM golang:1.26-alpine AS builder

WORKDIR /app

COPY go.mod go.sum ./
RUN go mod download

COPY . .

RUN go build -o auth-service ./cmd/server/main.go

FROM alpine:latest

WORKDIR /app

COPY --from=builder /app/auth-service .
COPY --from=builder /app/private.pem .
COPY --from=builder /app/public.pem .

EXPOSE 8000

CMD ["./auth-service"]