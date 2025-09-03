FROM ubuntu:22.04
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# --- START MODIFICATION: Package list updated for 5 languages ---
# Removed: racket, ocaml-interp, ocaml, r-base, lua5.3, lua-unit
# Added: luarocks (for Lua), software-properties-common (for R/OCaml PPAs)
RUN apt-get update -yqq && apt-get install -yqq --fix-missing \
    curl build-essential python3-pip python3-tqdm \
    wget \
    software-properties-common

# R (Updated Version)
# Use the official CRAN repository to get a more recent version of R
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    r-base r-base-dev && \
    rm -rf /var/lib/apt/lists/*


# Racket (non-interactive install)
RUN wget -nv https://download.racket-lang.org/installers/8.18/racket-8.18-x86_64-linux-buster-cs.sh -O racket-installer.sh && \
    sh racket-installer.sh --in-place --dest /usr/racket && \
    rm racket-installer.sh

ENV PATH="/usr/racket/bin:${PATH}"

# Install common Racket packages at installation scope
RUN raco pkg install --auto --scope installation crypto srfi math data

# OCaml (FIXED: System-wide installation for multi-user access)
ENV OPAMROOT=/opt/opam
RUN apt-get update && apt-get install -y --no-install-recommends opam && \
    # Initialize opam in the system-wide location defined by OPAMROOT
    opam init --disable-sandboxing --bare && \
    # Create the OCaml compiler switch
    opam switch create 5.2.1 && \
    # Ensure all users can read and execute the files
    chmod -R a+rX "$OPAMROOT" && \
    rm -rf /var/lib/apt/lists/*

# Set the PATH to the system-wide OCaml binaries.
# Note: opam automatically creates the switch inside the OPAMROOT dir.
ENV PATH="${OPAMROOT}/5.2.1/bin:${PATH}"


# Lua (FIXED)
# Install a recent Lua version and use the luarocks package manager for libraries
RUN apt-get update && apt-get install -yqq lua5.4 luarocks && \
    luarocks install luaunit && \
    rm -rf /var/lib/apt/lists/*

# --- END MODIFICATION ---

# Post-processing
# Julia (FIXED non-interactive install)
ARG JULIA_VERSION=1.11.6
RUN mkdir -p /opt/julia && \
    wget -nv "https://julialang-s3.julialang.org/bin/linux/x64/${JULIA_VERSION%.*}/julia-${JULIA_VERSION}-linux-x86_64.tar.gz" -O /tmp/julia.tar.gz && \
    tar -xzf /tmp/julia.tar.gz -C /opt/julia --strip-components=1 && \
    rm /tmp/julia.tar.gz
ENV PATH="/opt/julia/bin:${PATH}"
# --- END MODIFICATION ---



# install numpy for humanevalplus
RUN python3 -m pip install numpy

COPY src /code
WORKDIR /code
ENTRYPOINT ["python3", "main.py"]