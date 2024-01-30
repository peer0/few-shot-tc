# 실험 결과 확인 방법

1. codebert/code/experiment 경로에 있습니다.
   - code_complex : BERT 결과, bs=8
   - code_complex_codebert : CodeBERT 결과, bs=8
   - code_complex_cw : CodeBERT,train data에 대해 배치별 클래스 하나씩 꼭 포함하도록 수정한 결과, bs=7
   - cc_complex_cw_loss_soft : 
      - train_labeled_loader를 위한 evaluation_batch 추가
      - training_statistics.csv에 loss를 추가
      - confmat.png 수정(축레이블추가,레이블을 기본형태로 변경,confmat.png(norm제거버전)추가)
      - seed0만 실험, bs=7, labeling_mode = 'soft'
   - 추후 실험
      - 주석을 제거한 (원본데이터, synonym_aug)
2. 가장 최근 결과 확인 
   - codebert/code/experiment/code_complex_cw/log 에 code_complex_java 데이터로 fewshot1,3,5,10한 결과가 있습니다.
   - 이전 결과(codebert/code/experiment/code_complex_codebert/log 와 비교해보시면 됩니다.
   - summary.csv파일에서 dataset,n_labeled_per_class 컬럼을 통해 어떤데이터(python,java)를 사용했는지, 몇 shot의 결과인지 확인하실 수 있습니다.
   - 경향성 확인을 위해서는 training_statistics.csv를 확인해주세요.
   ex.codebert/code/experiment/code_complex_cw/log/20240126-163053/0/training_statistics.csv
