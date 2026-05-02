FROM nginx:alpine

# Copy only the HTML/CSS/JS files (not the Dockerfile)
COPY frontend/index.html /usr/share/nginx/html/

# Custom nginx config — note: using a separate file avoids shell variable
# interpolation issues with heredocs ($uri would be eaten by sh)
COPY frontend/nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 3000
