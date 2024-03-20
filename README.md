# Jointmatch 

1. origin : 기본 코드

2. codebert : bert대신 codebert적용한 코드와 실험

3. multiple : 서로 다른 모델 두 개, 세 개 사용할 수 있도록 수정한 코드와 실험


## 1. start
```
# 자신에 맞게 username, email 변경
git clone [레포주소]

git config --global user.name "inistory"

git config --global user.email "jungin3486@gmail.com"
```

```
pip3 install virtualenv 

virtualenv jointmatch --python=3.9
source jointmatch/bin/activate

pip install torch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1

pip install -r requirements.txt
```