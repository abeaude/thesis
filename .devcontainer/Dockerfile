FROM texlive/texlive:latest-full
RUN apt-get update && apt-get install -y \
    git perl unzip wget\
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/* \
    && wget https://github.com/githubnext/monaspace/releases/download/v1.101/monaspace-v1.101.zip \
    && unzip monaspace-v1.101.zip \
    && cd monaspace-v1.101 \
    && mkdir -p ~/.local/share/fonts \
    && rm -rf ~/.local/share/fonts/Monaspace* \
    && mkdir -p ~/.local/share/fonts/Monaspace/ \
    # copy all fonts from ./otf to ~/.local/share/fonts
    && cp ./fonts/otf/* ~/.local/share/fonts/Monaspace/ \
    # copy variable fonts from ./variable to ~/.local/share/fonts
    && cp ./fonts/variable/* ~/.local/share/fonts/Monaspace/ \
    # Build font information caches
    && fc-cache -f \
    && rm -rf monaspace-v1.101 monaspace-v1.101.zip

