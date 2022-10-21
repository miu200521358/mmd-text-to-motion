# mmd-text-to-motion

## Install

conda create -n mttm pip python=3.9
conda activate mttm
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
pip install git+https://github.com/openai/CLIP.git
pip install -r requirements.txt

`xxxxx/Anaconda3/envs/mttm/Lib/site-packages/bezier/extra-dll/bezier-2a44d276.dll` を ひとつ上の `bezier` フォルダに配置する
`xxxxx/Anaconda3/envs/mttm/Lib/site-packages/bezier/bezier-2a44d276.dll`

git submodule add https://github.com/miu200521358/motion-diffusion-model src/mdm
