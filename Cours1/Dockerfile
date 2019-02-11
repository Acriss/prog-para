FROM gcc:8.2.0

RUN apt update && \
    apt install -y zsh && \
    chsh -s /usr/bin/zsh

COPY zshrc.txt /etc/zsh/zshrc

ENV SHELL="/usr/bin/zsh"

CMD ["tail", "-f", "/dev/null"]
