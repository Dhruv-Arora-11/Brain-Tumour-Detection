FROM node:22-slim
RUN apt-get update && apt-get install -y python3 python3-pip --no-install-recommends && rm -rf /var/lib/apt/lists/* && ln -sf /usr/bin/python3 /usr/bin/python
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm install
COPY . .
RUN mkdir -p uploads
EXPOSE 5000
CMD ["node", "server.js"]
