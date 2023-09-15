#!/bin/bash

#read -p "Please Enter You mode(1~3): " MODE
#read -p "Wether use state predictor(1~3): " USP

#NAME_LIST="apex1 bc0 c1355 c5315 c6288 c7552 dalu i10 k2 mainpla" #
#NAME_LIST="c1355 c5315 c6288 c7552 dalu i10 k2 mainpla"
#NAME_LIST="log2 max multiplier sqrt square aes_secworks aes_xcrypt bp_be ethernet picosoc jpeg tinyRocket vga_lcd wb_conmax"
#NAME_LIST="apex1 bc0 c1355 c5315 c6288 c7552 dalu i10 k2 mainpla div log2 max multiplier sqrt square aes_secworks aes_xcrypt bp_be ethernet picosoc jpeg tinyRocket vga_lcd wb_conmax"
#NAME_LIST="sqrt square aes_secworks aes_xcrypt bp_be ethernet picosoc jpeg tinyRocket vga_lcd wb_conmax"
#NAME_LIST="bp_be picosoc jpeg vga_lcd"
#NAME_LIST="ethernet"
NAME_LIST="apex1"
#tested={tv80 mem_ctrl i2c  iir fir ac97_ctrl wb_dma des3 sha256 usb_phy sasc spi ss_pcm simple_spi fpu}
#tested={spi ss_pcm usb_phy sasc wb_dma simple_spi pci ac97_ctrl mem_ctrl des3_area sha256 fir iir tv80 dynamic_node}
#cuda_out_of_mem={ethernet aes jpeg}
#NAME_LIST="simple_spi pci ac97_ctrl mem_ctrl des3_area sha256 fir iir tv80 dynamic_node"
#NAME_LIST="pci"
#cd ./OPENABC2_DATASET/lp1 || exit

for NAME in $NAME_LIST
do
#python select_files.py --ip ${NAME}
python GPT-LS_experiment.py --design ${NAME} --design_file orig/${NAME}_orig.bench
#python /home/lcy/PycharmProjects/OpenABC/models/qor/SynthNetV1/train.py
done


