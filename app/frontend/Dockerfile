# --- Build Stage ---
FROM node:20-slim AS build

ARG VITE_API_URL=http://localhost:8000
ENV VITE_API_URL=$VITE_API_URL

WORKDIR /app
COPY app/frontend/package*.json ./
RUN npm ci
COPY app/frontend/ ./
RUN npx vite build

# --- Serve Stage ---
FROM nginx:alpine

COPY --from=build /app/dist /usr/share/nginx/html
COPY app/frontend/nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 5173
CMD ["nginx", "-g", "daemon off;"]
