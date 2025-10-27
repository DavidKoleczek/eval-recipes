# uv is already installed in the base.dockerfile

ENV PATH="/root/.cargo/bin:/root/.local/bin:$PATH"

RUN uv tool install git+https://github.com/microsoft/amplifier@next

RUN amplifier --version
