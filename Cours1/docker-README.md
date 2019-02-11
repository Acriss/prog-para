# Use the Docker files

When you are in this directory,

Build the image (if not already done)
```
docker build -t pp_gcc_img .
```

Run the container in background
```
docker-compose up -d
```

Attach to the container, quit with `^D`
```
docker exec -it pp_gcc zsh
```

Kill the container
```
docker-compose down
```
