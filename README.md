TODO Documentation

Output docker image

"""
docker save -o oqgn-instance.tar oqgn-instance
"""

Load image

"""
docker load -i oqgn-instance.tar
"""

Launch image

"""
docker run \
 --network none \
 --cpus="8" \
 --memory="8G" \
 --mount type=bind,source="$(pwd)/config.json",target=/var/www/config.json \
 --mount type=bind,source="$(pwd)/models/skwiz",target=/var/www/models/skwiz \
 --mount type=bind,source="$(pwd)/workspace/input",target=/var/www/input \
 --mount type=bind,source="$(pwd)/workspace/input_processed",target=/var/www/input_processed \
 --mount type=bind,source="$(pwd)/workspace/output",target=/var/www/output \
 --mount type=bind,source="$(pwd)/workspace/error",target=/var/www/error \
 oqgn-instance \
 python -m src.app
"""
