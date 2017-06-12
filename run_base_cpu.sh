#!/bin/sh

set -ex

docker run -v /usr/local/share/datasets:/usr/local/share/datasets -it x/docker/tensorflow:1.1.0-cpu bash
