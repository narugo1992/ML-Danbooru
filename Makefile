.PHONY: export

PYTHON     ?= $(shell which python)

DIST_DIR   ?= dist

IMAGE_FILE ?=
CKPT_NAME  ?= TResnet-D-FLq_ema_6-10000.ckpt
ONNX_FILE  ?= ${DIST_DIR}/$(basename ${CKPT_NAME}).onnx
HALF       ?=

export:
	$(PYTHON) export.py \
	  --model_name tresnet_d \
	  --num_of_groups 32 \
	  --ckpt_online ${CKPT_NAME} \
	  --image_size 512 --keep_ratio yes \
	  $(if ${HALF},--fp16,) \
  	  --data "${IMAGE_FILE}" \
  	  --onnx_file "${ONNX_FILE}"

clean:
	rm -rf "${DIST_DIR}"