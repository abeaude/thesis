FROM texlive/texlive:latest-full
ENV LATEXINDENT_CONFIG="/root/"
RUN apt-get update && apt-get install -y \
    git perl unzip wget ssh inkscape libnotify4 libxss1 xdg-utils libsecret-1-0 libxml2 pre-commit\
    && wget -q https://github.com/jgraph/drawio-desktop/releases/download/v24.5.3/drawio-amd64-24.5.3.deb \
    && dpkg -i drawio-amd64-24.5.3.deb \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/* \
    && rm -rf drawio-amd64-24.5.3.deb \
    && wget -q https://github.com/githubnext/monaspace/releases/download/v1.101/monaspace-v1.101.zip \
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
