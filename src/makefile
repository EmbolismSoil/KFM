
SRC_DIR := ./
all: model.pb.cc model.pb.h

model.pb.cc model.pb.h: ${SRC_DIR}/model.proto
	protoc -I${SRC_DIR} --cpp_out=./ ${SRC_DIR}/model.proto

.PHONY:clean

clean:
	rm model.pb.cc model.pb.h
