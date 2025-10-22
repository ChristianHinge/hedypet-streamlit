#!/bin/bash

# Initialize Let's Encrypt certificates for hedypet.depict.dk

domains=(hedypet.depict.dk)
email="christiahn.hinge@dregionh.dk" # Change this to your email
staging=0 # Set to 1 for testing

echo "### Preparing directories for certbot..."
mkdir -p certbot/conf certbot/www

echo "### Requesting Let's Encrypt certificate for ${domains[0]}..."

if [ $staging != "0" ]; then staging_arg="--staging"; fi

docker compose run --rm certbot certonly --webroot \
    --webroot-path=/var/www/certbot \
    $staging_arg \
    --email $email \
    --agree-tos \
    --no-eff-email \
    -d ${domains[0]}

echo "### Reloading nginx..."
docker compose exec nginx nginx -s reload
