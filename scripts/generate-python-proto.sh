#!/usr/bin/env bash
protoc --proto_path=protos/ --python_out=python_generated/ protos/message.proto
