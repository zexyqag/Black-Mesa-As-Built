# bmab in a container: the app (the default; publish its port on 127.0.0.1 only), or any other bmab command, such as importing
# your games. Game files are never in the image: mount them (/bmab/data, /bmab/out, /bmab/cache, /redlines; settings in
# bmab/redlines.py and bmab/github.py).
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/* \
    && git config --system --add safe.directory '*'
WORKDIR /bmab
COPY pyproject.toml ./
RUN pip install --no-cache-dir numpy pillow scipy
COPY sources.json sources.lock.json ./
COPY bmab ./bmab
RUN mkdir -p data out cache downloads /redlines && chmod 777 data out cache downloads /redlines  # mounted over (a new volume takes these rights), or written by whichever user it runs as
ENV BMAB_REDLINES=/redlines PYTHONUNBUFFERED=1 HOME=/tmp
EXPOSE 8650
ENTRYPOINT ["python3", "-m", "bmab"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8650"]
