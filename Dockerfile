FROM jupyter/scipy-notebook

COPY --chown=jovyan:users . /home/jovyan/work
