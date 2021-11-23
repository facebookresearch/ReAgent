#!/bin/bash
rm -rf api/* && rm -rf ~/github/HorizonDocs && sphinx-build -b html -E -v . ~/github/HorizonDocs
