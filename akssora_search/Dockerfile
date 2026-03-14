FROM oven/bun:1 AS base

WORKDIR /app

# Copy dependency files first (better caching)
COPY package.json bun.lock ./

# Install dependencies
RUN bun install

# Copy rest of project
COPY . .

# Build Next.js app
RUN bun run build

# Production stage
FROM oven/bun:1

WORKDIR /app

COPY --from=base /app /app

EXPOSE 3000

CMD ["bun", "run", "start"]