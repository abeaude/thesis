FROM texlive/texlive@sha256:fdf2fbe837577ccf354a5605701ccc1853f895366cfcdf9c7859a872f10f6822
COPY PygmentsFlexoki PygmentsFlexoki
RUN apt-get update && apt-get install -y --no-install-recommends \
    git perl unzip wget ssh inkscape libnotify4 libxss1 xdg-utils libsecret-1-0 libxml2 pre-commit python3-venv pandoc curl gdebi-core \
    && wget -q https://github.com/jgraph/drawio-desktop/releases/download/v24.5.3/drawio-amd64-24.5.3.deb \
    && dpkg -i drawio-amd64-24.5.3.deb \
    && rm -rf /var/lib/apt/lists/* \
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
    && apt-get clean \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /var/cache/apt/ \
    && rm -rf /tmp/* \
    && rm -rf drawio-amd64-24.5.3.deb \
    && rm -rf monaspace-v1.101 monaspace-v1.101.zip \
    && python -m venv /root/venv/ \
    && cd /workdir/PygmentsFlexoki \
    && /root/venv/bin/pip install . \
    && echo "main_memory = 7999999" >> /usr/local/texlive/2024/texmf.cnf \
    && fmtutil-sys --all
RUN curl -LO https://quarto.org/download/latest/quarto-linux-amd64.deb
RUN gdebi --non-interactive quarto-linux-amd64.deb