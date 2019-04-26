# Sample Project実行(NN)
* NNCにログイン
* Project画面を表示し、"tutorial.basics.01_logistic_regression"を選択。
* "You can create a new project from this sample project. Enter a new project name between 1 and 255 characters log."と表示されるので、"tutorial.basics.01_logistic_regression-trial"として、OKを選択。
* 作成したプロジェクトがリストされるので、選択すると、EDIT画面に遷移する。
* EDIT画面には既に作成済みのニューラルネットワークが表示される。
* ここではSmall MNISTを学習データセットとして使用する。
* これはMNISTから一部を抽出したもので、手書き数字0～9が描かれたグレースケール画像の集まりで画像一枚のサイズは28×28px。
* DATASETタブを選択し、Small MNISTを確認できる。
* 学習データ(mnist.small_mnist_4or9_training)及びテストデータ(mnist.small_mnist_4or9_test)はそれぞれ左側のタブで切り替えて確認ができる。
* 各データは以下の要素で構成されている。
    * index：画像番号(連番)
    * x:image：画像データ
    * y:label：0 or 1(画像が4の場合は0、9の場合は1)
* また画面上部には学習・テスト時の設定の有無が表示されており、有効にする場合はチェックを入れる。
    * Shuffle：学習時にデータセットをシャッフルする。
    * Image Normalization(1.0/255.0)：ピクセル値を1/225にして値を0.0～1.0にする。
* EDITタブを選択し、作成済みネットワークを表示し、構成を確認する。
* 作成済みネットワークは大きく入力層(I)と出力層(A・S・B)から成る。
* コンポーネントを選択すると、左下のLayer Propertyにレイヤの詳細設定が表示される。
* また各層の右側にはその層の入力サイズが表示されている。
* 次にCONFIGタブを選択し、学習条件を確認する。
* 左側のタブをそれぞれ選択し、各設定を確認・変更する
    * Global Config：学習の全体的な条件を設定
    * Optimizer：重みの更新に関する設定
    * train_error：学習時の誤差計算に関する設定
    * valid_error：評価(テスト)時の誤差計算に関する設定
    * Executor：モデルの評価に関する設定
* この時点ではデフォルトの設定から変更はしないままとなる。
* 次にEDITタブを選択し、右側のRunボタンを選択することで学習を実行する。
* TRAINING画面に遷移し、学習が始まる。
* 以下のようなログが出力され、学習が終了する。
    ```
    2019-03-11 04:50:40,965 [worker]: [INFO]: download: 13612/configurations/44876/data.sdcproj to work/data.sdcproj
    2019-03-11 04:50:41,788 [worker]: [INFO]: sdeep_console_cli_util create_prototxt -i "/home/nnabla/work/data.sdcproj" -o "/home/nnabla/work/network.prototxt" -p "/home/nnabla/work/param_assign.csv"
    2019-03-11 04:50:41,965 [worker]: [INFO]: sdeep_console_cli_util create_result_ini -i "/home/nnabla/work/data.sdcproj" -y "/home/nnabla/empty_monitoring_report.yml" -o "/home/nnabla/work/result.ini"
    2019-03-11 04:50:42,101 [worker]: [INFO]: nnabla_cli train -c /home/nnabla/work/network.prototxt -o /home/nnabla/results -s /home/nnabla/work/data.sdcproj -a /home/nnabla/work/param_assign.csv
    2019-03-11 04:50:43,783 [nnabla]: Train with contexts ['cpu']
    2019-03-11 04:50:43,829 [nnabla]: Training epoch 1 of 100 begin
    2019-03-11 04:50:44,134 [nnabla]: epoch 1 of 100 cost=0.675991  {train_error=0.603888, valid_error=0.615436} time=(0.1s /13.8s)
    2019-03-11 04:50:44,222 [nnabla]: epoch 2 of 100 cost=0.560800  {train_error=0.517158, valid_error=0.544698} time=(0.3s /17.2s)
    2019-03-11 04:50:44,320 [nnabla]: epoch 3 of 100 cost=0.483067  {train_error=0.453852, valid_error=0.494678} time=(0.4s /14.4s)
    2019-03-11 04:50:44,427 [nnabla]: epoch 4 of 100 cost=0.430934  {train_error=0.401537, valid_error=0.451036} time=(0.5s /13.5s)
    2019-03-11 04:50:44,533 [nnabla]: epoch 5 of 100 cost=0.383878  {train_error=0.360693, valid_error=0.413800} time=(0.6s /12.9s)
    2019-03-11 04:50:44,582 [nnabla]: epoch 6 of 100 cost=0.348642  time=(0.8s /12.5s)
    2019-03-11 04:50:44,629 [nnabla]: epoch 7 of 100 cost=0.322732  time=(0.8s /11.4s)
    2019-03-11 04:50:44,677 [nnabla]: epoch 8 of 100 cost=0.296783  time=(0.8s /10.6s)
    2019-03-11 04:50:44,724 [nnabla]: epoch 9 of 100 cost=0.280102  time=(0.9s /9.9s)
    2019-03-11 04:50:44,837 [nnabla]: epoch 10 of 100 cost=0.254238  {train_error=0.253839, valid_error=0.312850} time=(1.0s /9.5s)
    2019-03-11 04:50:44,883 [nnabla]: epoch 11 of 100 cost=0.249883  time=(1.1s /9.6s)
    2019-03-11 04:50:44,936 [nnabla]: epoch 12 of 100 cost=0.222481  time=(1.1s /9.2s)
    ・・・
    2019-03-11 04:50:51,931 [nnabla]: epoch 89 of 100 cost=0.060362  time=(8.1s /9.1s)
    2019-03-11 04:50:52,021 [nnabla]: epoch 90 of 100 cost=0.064903  {train_error=0.064058, valid_error=0.127886} time=(8.1s /9.0s)
    2019-03-11 04:50:52,063 [nnabla]: epoch 91 of 100 cost=0.067806  time=(8.2s /9.0s)
    2019-03-11 04:50:52,103 [nnabla]: epoch 92 of 100 cost=0.062350  time=(8.3s /9.0s)
    2019-03-11 04:50:52,144 [nnabla]: epoch 93 of 100 cost=0.066395  time=(8.3s /8.9s)
    2019-03-11 04:50:52,185 [nnabla]: epoch 94 of 100 cost=0.058158  time=(8.4s /8.9s)
    2019-03-11 04:50:52,226 [nnabla]: epoch 95 of 100 cost=0.063852  time=(8.4s /8.8s)
    2019-03-11 04:50:52,266 [nnabla]: epoch 96 of 100 cost=0.066665  time=(8.4s /8.8s)
    2019-03-11 04:50:52,307 [nnabla]: epoch 97 of 100 cost=0.057507  time=(8.5s /8.7s)
    2019-03-11 04:50:52,348 [nnabla]: epoch 98 of 100 cost=0.062721  time=(8.5s /8.7s)
    2019-03-11 04:50:52,389 [nnabla]: epoch 99 of 100 cost=0.063643  time=(8.6s /8.6s)
    2019-03-11 04:50:52,480 [nnabla]: epoch 100 of 100 cost=0.059969  {train_error=0.060852, valid_error=0.118805} time=(8.6s /8.6s)
    2019-03-11 04:50:52,480 [nnabla]: Training Completed.
    NNabla command line interface (Version 1.0.5.console_day3-fix-181220, Build 181219104847)
    2019-03-11 04:50:57,154 [worker]: [INFO]: create result_train.nnp
    2019-03-11 04:50:58,579 [worker]: [INFO]: create result.nnb
    2019-03-11 04:51:03,624 [worker]: [INFO]: create result.onnx
    2019-03-11 04:51:06,777 [worker]: [INFO]: worker done
    ```
* また学習カーブとして、以下のグラフが表示される。
    * cost
    * Training Error(学習誤差曲線)
    * validation Error(評価誤差曲線)
* 画面左側には過去の学習結果が一覧で表示され、学習結果を比較できる。
* 画面上部には学習の進捗情報が表示され、Elapsedで学習開始からの経過時間、Remainingで現在からの学習終了予測時間、TotalでElapsed + Remainingの合計時間が表示される。
* 次に作成したモデルの評価を実行する。
* TRAININGタブ右側のRunボタンを選択することでモデルの評価が開始される。
* EVALUATION画面に遷移し、モデルの評価が始まる。
* 以下のようなログが出力され、評価が終了する。
    ```
    2019-03-11 05:44:36,130 [worker]: [INFO]: download: 13612/configurations/44876/data.sdcproj to work/data.sdcproj
    2019-03-11 05:44:36,884 [worker]: [INFO]: Find learned parameter file: results_current_100.nnp
    2019-03-11 05:44:38,664 [worker]: [INFO]: download: 13612/results/44876/result.nnp to results/result.nnp
    2019-03-11 05:44:39,564 [worker]: [INFO]: download: 13612/results/44876/results_best_100.nnp to results/results_best_100.nnp
    2019-03-11 05:44:40,419 [worker]: [INFO]: download: 13612/results/44876/results_current_100.nnp to results/results_current_100.nnp
    2019-03-11 05:44:40,425 [worker]: [INFO]: Use config file: 13612/results/44876/results_current_100.nnp, instead of config file: 13612/configurations/44876/data.sdcproj
    2019-03-11 05:44:40,470 [worker]: [INFO]: nnabla_cli forward -c /home/nnabla/results/results_current_100.nnp -d /dataset-cache/ccbf15a0-bcb6-4ba6-b10e-27fc877c4348/2.cache -o /home/nnabla/results
    2019-03-11 05:44:42,202 [nnabla]: data 64 / 500
    2019-03-11 05:44:42,210 [nnabla]: data 128 / 500
    2019-03-11 05:44:42,217 [nnabla]: data 192 / 500
    2019-03-11 05:44:42,225 [nnabla]: data 256 / 500
    2019-03-11 05:44:42,232 [nnabla]: data 320 / 500
    2019-03-11 05:44:42,240 [nnabla]: data 384 / 500
    2019-03-11 05:44:42,245 [nnabla]: data 448 / 500
    2019-03-11 05:44:42,249 [nnabla]: data 500 / 500
    2019-03-11 05:44:42,273 [nnabla]: Add output_result.zip to result.nnp.
    2019-03-11 05:44:42,273 [nnabla]: Add output_result.zip to results_current_100.nnp.
    2019-03-11 05:44:42,274 [nnabla]: Add output_result.zip to results_best_100.nnp.
    2019-03-11 05:44:42,275 [nnabla]: Forward Completed.
    NNabla command line interface (Version 1.0.5.console_day3-fix-181220, Build 181219104847)
    2019-03-11 05:44:43,189 [worker]: [INFO]: upload: results/output_result.csv to 13612/results/44876/output_result.csv
    2019-03-11 05:44:43,195 [worker]: [INFO]: create confusion_matrix.json
    2019-03-11 05:44:43,267 [worker]: [INFO]: confusion_matrix.json created
    2019-03-11 05:44:44,189 [worker]: [INFO]: upload: work/confusion_matrix.json to 13612/results/44876/confusion_matrix.json
    2019-03-11 05:44:45,316 [worker]: [INFO]: upload: results/result.nnp to 13612/results/44876/result.nnp
    2019-03-11 05:44:46,199 [worker]: [INFO]: upload: results/results_best_100.nnp to 13612/results/44876/results_best_100.nnp
    2019-03-11 05:44:46,921 [worker]: [INFO]: upload: results/results_current_100.nnp to 13612/results/44876/results_current_100.nnp
    2019-03-11 05:44:46,929 [worker]: [INFO]: create result_evaluate.nnp
    2019-03-11 05:44:55,284 [worker]: [INFO]: Completed to pageing file creator.
    2019-03-11 05:44:55,291 [worker]: [INFO]: worker done
    ```
* また評価結果として、以下を選択して表示することができる。
    * Output Result
        * Indexと評価データイメージ、ラベル値とそのモデルにおける出力結果(y')が表示される。
    * Confusion Matrix
        * 混同行列(分類問題において正しく分類できたデータ数とできなかったデータ数をまとめた表)とモデルの精度などの情報が表示される。
        * つまりここではy=0の正解ラベルに対し、y'=0, 1とモデルが判別したデータ数、y=1の正解ラベルに対し、y'=0, 1とモデルが判別したデータ数の表となる。
        * この表から算出できるRecall(再現率)と適合率(Precision)、精度(Accuracy)も表示される。
        * Recallは正解データから見た正解と出力結果の一致割合のことである。
        * Precisionは出力結果から見た正解と出力結果の一致割合のことである。
        * Accuracyは全ての評価データのうち、正しく分類できた割合のことである。
            |     | y'=0 | y'=1 |
            |:---:|:----:|:----:|
            | y=0 | 237  | 13   |
            | y=1 | 12   | 238  |
        * Confusion Matrixが上記とすると、Recall/Precision/Accuracyはそれぞれ以下の通りとなる。
            * Recall：
                * 4の画像に対するRecall：237÷(237+13) = 237÷250 ≒ 0.948
                * 9の画像に対するRecall：238÷(238+12) = 238÷250 ≒ 0.952
            * Precision：
                * 4の画像に対するPrecision：237÷(237+12) = 237÷249 ≒ 0.9518
                * 9の画像に対するPrecision：238÷(238+13) = 238÷251 ≒ 0.9482
            * Accuraccy：
                * 作成モデルのAccuracy：(237+238)÷(237+238+12+13) = 475÷500 = 0.95
* 画面左側/上部にはTRAININGタブと同様、過去の学習結果一覧・及び学習の進捗情報が表示される。
* 画面上を右クリックすることでSave CSV as...のコンテキストが表示され、CSV形式でConfusionMatrixを保存することが可能。


# Sample Project実行(CNN)
* NNCにログイン
* Project画面を表示し、"tutorial.basics.02_binary_cnn"を選択。
* "You can create a new project from this sample project. Enter a new project name between 1 and 255 characters log."と表示されるので、"tutorial.basics.02_binary_cnn-trial"として、OKを選択。
* 作成したプロジェクトがリストされるので、選択すると、EDIT画面に遷移する。
* EDIT画面には既に作成済みのニューラルネットワークが表示される。
* ここでもNNの場合と同様、はSmall MNISTを学習データセットとして使用する。
* DATASETタブを選択し、Small MNISTを確認できる。学習・評価データ共にNNと同じものを使用する。
* EDITタブを選択し、作成済みネットワークを表示し、構成を確認する。
* 作成済みネットワークは大きく入力層(I)と畳み込み層1(C・M・T)、畳み込み層2(C・M・T)、全結合層(A・T)、出力層(A・S・B)から成る。
* 次にCONFIGタブを選択し、学習条件を確認する。
* NNの場合と設定画面は同様で、ここでは設定は変更せず、デフォルト設定のままとする。
* 次にEDITタブを選択し、右側のRunボタンを選択することで学習を実行する。
* TRAINING画面に遷移し、学習が始まる。
* 以下のようなログが出力され、学習が終了する。
    ```
    2019-03-12 07:17:08,503 [worker]: [INFO]: download: 13631/configurations/44916/data.sdcproj to work/data.sdcproj
    2019-03-12 07:17:09,042 [worker]: [INFO]: sdeep_console_cli_util create_prototxt -i "/home/nnabla/work/data.sdcproj" -o "/home/nnabla/work/network.prototxt" -p "/home/nnabla/work/param_assign.csv"
    2019-03-12 07:17:09,137 [worker]: [INFO]: sdeep_console_cli_util create_result_ini -i "/home/nnabla/work/data.sdcproj" -y "/home/nnabla/empty_monitoring_report.yml" -o "/home/nnabla/work/result.ini"
    2019-03-12 07:17:09,240 [worker]: [INFO]: nnabla_cli train -c /home/nnabla/work/network.prototxt -o /home/nnabla/results -s /home/nnabla/work/data.sdcproj -a /home/nnabla/work/param_assign.csv
    2019-03-12 07:17:10,120 [nnabla]: Train with contexts ['cpu']
    2019-03-12 07:17:10,156 [nnabla]: Training epoch 1 of 100 begin
    2019-03-12 07:17:11,534 [nnabla]: epoch 1 of 100 cost=0.500818  {train_error=0.352672, valid_error=0.391087} time=(0.8s /78.6s)
    2019-03-12 07:17:12,642 [nnabla]: epoch 2 of 100 cost=0.286012  {train_error=0.215833, valid_error=0.256652} time=(2.0s /101.9s)
    2019-03-12 07:17:13,743 [nnabla]: epoch 3 of 100 cost=0.183753  {train_error=0.158449, valid_error=0.205073} time=(3.1s /104.9s)
    2019-03-12 07:17:15,728 [nnabla]: epoch 4 of 100 cost=0.134087  {train_error=0.118959, valid_error=0.156219} time=(4.9s /121.8s)
    2019-03-12 07:17:17,389 [nnabla]: epoch 5 of 100 cost=0.105694  {train_error=0.089929, valid_error=0.122632} time=(6.7s /133.9s)
    2019-03-12 07:17:18,058 [nnabla]: epoch 6 of 100 cost=0.086228  time=(7.9s /131.5s)
    2019-03-12 07:17:18,727 [nnabla]: epoch 7 of 100 cost=0.076900  time=(8.6s /122.3s)
    2019-03-12 07:17:19,395 [nnabla]: epoch 8 of 100 cost=0.060771  time=(9.2s /115.4s)
    2019-03-12 07:17:20,061 [nnabla]: epoch 9 of 100 cost=0.061615  time=(9.9s /110.0s)
    2019-03-12 07:17:21,520 [nnabla]: epoch 10 of 100 cost=0.046809  {train_error=0.043224, valid_error=0.083557} time=(10.9s /108.5s)
    2019-03-12 07:17:22,229 [nnabla]: epoch 11 of 100 cost=0.043935  time=(12.1s /109.7s)
    2019-03-12 07:17:23,282 [nnabla]: epoch 12 of 100 cost=0.036136  time=(13.1s /109.3s)
    ・・・
    2019-03-12 07:18:32,290 [nnabla]: epoch 88 of 100 cost=0.000724  time=(82.1s /93.3s)
    2019-03-12 07:18:32,958 [nnabla]: epoch 89 of 100 cost=0.000680  time=(82.8s /93.0s)
    2019-03-12 07:18:34,077 [nnabla]: epoch 90 of 100 cost=0.000683  {train_error=0.000680, valid_error=0.051400} time=(83.5s /92.7s)
    2019-03-12 07:18:35,158 [nnabla]: epoch 91 of 100 cost=0.000686  time=(85.0s /93.4s)
    2019-03-12 07:18:36,730 [nnabla]: epoch 92 of 100 cost=0.000635  time=(86.6s /94.1s)
    2019-03-12 07:18:37,793 [nnabla]: epoch 93 of 100 cost=0.000657  time=(87.6s /94.2s)
    2019-03-12 07:18:38,458 [nnabla]: epoch 94 of 100 cost=0.000616  time=(88.3s /93.9s)
    2019-03-12 07:18:39,119 [nnabla]: epoch 95 of 100 cost=0.000616  time=(89.0s /93.6s)
    2019-03-12 07:18:39,782 [nnabla]: epoch 96 of 100 cost=0.000617  time=(89.6s /93.4s)
    2019-03-12 07:18:40,447 [nnabla]: epoch 97 of 100 cost=0.000568  time=(90.3s /93.1s)
    2019-03-12 07:18:41,116 [nnabla]: epoch 98 of 100 cost=0.000581  time=(91.0s /92.8s)
    2019-03-12 07:18:41,820 [nnabla]: epoch 99 of 100 cost=0.000573  time=(91.7s /92.6s)
    2019-03-12 07:18:43,265 [nnabla]: epoch 100 of 100 cost=0.000545  {train_error=0.000550, valid_error=0.035466} time=(92.7s /92.7s)
    2019-03-12 07:18:43,265 [nnabla]: Training Completed.
    NNabla command line interface (Version 1.0.5.console_day3-fix-181220, Build 181219104847)
    2019-03-12 07:18:46,092 [worker]: [INFO]: create result_train.nnp
    2019-03-12 07:18:47,220 [worker]: [INFO]: create result.nnb
    2019-03-12 07:18:50,236 [worker]: [INFO]: create result.onnx
    2019-03-12 07:18:52,548 [worker]: [INFO]: worker done
    ```
* またNNの場合と同様、学習カーブとして、以下のグラフが表示される。
    * cost
    * Training Error(学習誤差曲線)
    * validation Error(評価誤差曲線)
* グラフからNNの場合と比べ、早いepochで誤差が収束していることが分かる。
* 画面左側の過去の学習結果が一覧、画面上部には学習の進捗情報の表示はNNの場合と同じ。
* 次に作成したモデルの評価を実行する。
* TRAININGタブ右側のRunボタンを選択することでモデルの評価が開始される。
* EVALUATION画面に遷移し、モデルの評価が始まる。
* 以下のようなログが出力され、評価が終了する。
    ```
    2019-03-12 07:27:32,819 [worker]: [INFO]: download: 13631/configurations/44916/data.sdcproj to work/data.sdcproj
    2019-03-12 07:27:33,388 [worker]: [INFO]: Find learned parameter file: results_current_100.nnp
    2019-03-12 07:27:34,460 [worker]: [INFO]: download: 13631/results/44916/result.nnp to results/result.nnp
    2019-03-12 07:27:35,082 [worker]: [INFO]: download: 13631/results/44916/results_best_100.nnp to results/results_best_100.nnp
    2019-03-12 07:27:35,706 [worker]: [INFO]: download: 13631/results/44916/results_current_100.nnp to results/results_current_100.nnp
    2019-03-12 07:27:35,710 [worker]: [INFO]: Use config file: 13631/results/44916/results_current_100.nnp, instead of config file: 13631/configurations/44916/data.sdcproj
    2019-03-12 07:27:35,751 [worker]: [INFO]: nnabla_cli forward -c /home/nnabla/results/results_current_100.nnp -d /dataset-cache/ccbf15a0-bcb6-4ba6-b10e-27fc877c4348/2.cache -o /home/nnabla/results
    2019-03-12 07:27:37,264 [nnabla]: data 64 / 500
    2019-03-12 07:27:37,284 [nnabla]: data 128 / 500
    2019-03-12 07:27:37,306 [nnabla]: data 192 / 500
    2019-03-12 07:27:37,327 [nnabla]: data 256 / 500
    2019-03-12 07:27:37,347 [nnabla]: data 320 / 500
    2019-03-12 07:27:37,368 [nnabla]: data 384 / 500
    2019-03-12 07:27:37,386 [nnabla]: data 448 / 500
    2019-03-12 07:27:37,401 [nnabla]: data 500 / 500
    2019-03-12 07:27:37,414 [nnabla]: Add output_result.zip to result.nnp.
    2019-03-12 07:27:37,414 [nnabla]: Add output_result.zip to results_current_100.nnp.
    2019-03-12 07:27:37,415 [nnabla]: Add output_result.zip to results_best_100.nnp.
    2019-03-12 07:27:37,415 [nnabla]: Forward Completed.
    NNabla command line interface (Version 1.0.5.console_day3-fix-181220, Build 181219104847)
    2019-03-12 07:27:38,118 [worker]: [INFO]: upload: results/output_result.csv to 13631/results/44916/output_result.csv
    2019-03-12 07:27:38,123 [worker]: [INFO]: create confusion_matrix.json
    2019-03-12 07:27:38,176 [worker]: [INFO]: confusion_matrix.json created
    2019-03-12 07:27:38,873 [worker]: [INFO]: upload: work/confusion_matrix.json to 13631/results/44916/confusion_matrix.json
    2019-03-12 07:27:39,420 [worker]: [INFO]: upload: results/result.nnp to 13631/results/44916/result.nnp
    2019-03-12 07:27:39,942 [worker]: [INFO]: upload: results/results_best_100.nnp to 13631/results/44916/results_best_100.nnp
    2019-03-12 07:27:40,546 [worker]: [INFO]: upload: results/results_current_100.nnp to 13631/results/44916/results_current_100.nnp
    2019-03-12 07:27:40,550 [worker]: [INFO]: create result_evaluate.nnp
    2019-03-12 07:27:46,681 [worker]: [INFO]: Completed to pageing file creator.
    2019-03-12 07:27:46,685 [worker]: [INFO]: worker done
    ```
* NNの場合と同様、評価結果として以下を選択して表示することができる。
    * Output Result
    * Confusion Matrix
* 画面左側/上部には過去の学習結果一覧・及び学習の進捗情報が表示される。
* 画面上を右クリックすることでSave CSV as...のコンテキストが表示され、CSV形式でConfusionMatrixを保存することが可能。
* また、Recall/Precision/Accuracyはそれぞれ以下の通りとなる。
    * Recall：
        * 4の画像に対するRecall：246÷(246+4) = 246÷250 ≒ 0.984
        * 9の画像に対するRecall：246÷(246+4) = 246÷250 ≒ 0.984
    * Precision：
        * 4の画像に対するPrecision：246÷(246+4) = 237÷250 ≒ 0.984
        * 9の画像に対するPrecision：238÷(246+4) = 238÷250 ≒ 0.984
    * Accuraccy：
        * 作成モデルのAccuracy：(246+246)÷(246+246+4+4) = 492÷500 = 0.984
* 上記の結果はNNの場合と比べ、より高い精度を得ることができていることが分かる。


# Sample Project実行(Iris)
* これまでは手書き文字画像という非構造化データの分類を行なってきたが、以下ではあやめの花の構造化データをデータセットとして用い、分類を行なう。
* Project画面を表示し、"tutorial.basics.01_logistic_regression-trial"を選択。
* EDIT画面に遷移したら、右上の"Save as"アイコンを選択し、プロジェクトを別名保存する。
* ここでは、"tutorial.basics.01_logistic_regression-trial3"とする。
* プロジェクトの別名保存には少し時間がかかるため、Project画面でstatusを確認する。
* Statusが"Saving"から変わるので、しばらく待つ。
* Dataset画面を表示すると、既に利用できるデータセットの一覧が確認できる。
* ここでは、学習データ、評価データとして以下を用いる。
    * 学習データ：iris_flower_dataset.iris_flower_dataset_training_delo
    * 評価データ：iris_flower_dataset.iris_flower_dataset_validation_delo
* データセット名を選択すると、画像の連番と画像ごとに以下の情報が表示される。
    * x_0:Sepal length：がくの長さ
    * x_1:Sepal width：がくの幅
    * x_2:Petal length：花びらの長さ
    * x_3:Petal width：花びらの幅
    * y:label：0～2の数値(アヤメの花の種類に対応)
        * 0：Setosa
        * 1：Versicolor
        * 2：Virginca
* 生成したプロジェクトのEDIT画面に移動し、DATASETタブを選択する。
* 画面上部にはTRAINING、VALIDATIONそれぞれで使用されているデータセットが表示されている。
* このデータセットのリンクを選択し、それぞれ上述のデータセットを設定する。
* 設定するとデータセットの内容が表示されることを確認する。
* 次にEDIT画面に移動し、作成済みのネットワークを変更する。
* ここでは以下のような全結合ニューラルネットワークを作成する。
    * 入力層：4ノード(x_0～x_3)
    * 中間層：25ノード, 活性化関数ReLU
    * 出力層：3ノード, 活性化関数Softmax
    * 誤差関数：多値分類交差エントロピー
* ネットワーク作成時の各種操作は以下の通り。
    * 足りないLayerの追加：
        * 左側のComponentsの追加可能Layer一覧からドラッグ&ドロップし、Mainタブ中に配置。
    * Layer間の結線：
        * Layer上下の黒丸同士をマウスで接続
    * Layer間の結線解除：
        * 結線を選択し、右クリック→Delete
    * 複数のLayerをまとめて操作：
        * ctrl押しながらLayerを選択
* ネットワークを作成・構成したら、各Layerのパラメータを以下の通り設定する。
    * 入力層のInput：4
    * 中間層のAffineのOutShape：25
    * 出力層のAffineのOutShape：25
* 各LayerのパラメータはLayerを選択し、左下のLayer Propertyで変更する。
* 前層のパラメータを変更すると、それが後段の層に関係するパラメータである場合、自動で後段の設定も変更される。
    * 入力層のInputを変更すると、AffineのInputも4になる。
    * AffineのOutShapeを25にすると、ReLU及び出力層のAffineのInputも25に変更される。
* 次にCONFIGタブを選択し、学習条件の設定を変更する。
* ここでは、Global Config画面でBatch Sizeを64→16に変更する
* EDITタブに戻り、右側のRunボタンを選択することで学習を実行すると、TRAINING画面に遷移し、学習が始まる。
* 以下のようなログが出力され、学習が終了する。
    ```
    2019-03-22 14:25:21,317 [worker]: [INFO]: download: 13954/configurations/45896/data.sdcproj to work/data.sdcproj
    2019-03-22 14:25:21,855 [worker]: [INFO]: sdeep_console_cli_util create_prototxt -i "/home/nnabla/work/data.sdcproj" -o "/home/nnabla/work/network.prototxt" -p "/home/nnabla/work/param_assign.csv"
    2019-03-22 14:25:21,941 [worker]: [INFO]: sdeep_console_cli_util create_result_ini -i "/home/nnabla/work/data.sdcproj" -y "/home/nnabla/empty_monitoring_report.yml" -o "/home/nnabla/work/result.ini"
    2019-03-22 14:25:22,038 [worker]: [INFO]: nnabla_cli train -c /home/nnabla/work/network.prototxt -o /home/nnabla/results -s /home/nnabla/work/data.sdcproj -a /home/nnabla/work/param_assign.csv
    2019-03-22 14:25:22,879 [nnabla]: Train with contexts ['cpu']
    2019-03-22 14:25:22,898 [nnabla]: Training epoch 1 of 100 begin
    2019-03-22 14:25:22,926 [nnabla]: epoch 1 of 100 cost=1.533403  {train_error=1.187518, valid_error=1.617892} time=(0.0s /1.2s)
    2019-03-22 14:25:22,944 [nnabla]: epoch 2 of 100 cost=1.271012  {train_error=1.061799, valid_error=0.801447} time=(0.0s /1.8s)
    2019-03-22 14:25:22,959 [nnabla]: epoch 3 of 100 cost=0.963418  {train_error=1.091719, valid_error=1.292670} time=(0.1s /1.8s)
    2019-03-22 14:25:22,974 [nnabla]: epoch 4 of 100 cost=1.048303  {train_error=0.996113, valid_error=0.850306} time=(0.1s /1.7s)
    2019-03-22 14:25:22,991 [nnabla]: epoch 5 of 100 cost=1.005608  {train_error=1.010743, valid_error=1.084535} time=(0.1s /1.7s)
    2019-03-22 14:25:22,999 [nnabla]: epoch 6 of 100 cost=0.982661  time=(0.1s /1.7s)
    2019-03-22 14:25:23,007 [nnabla]: epoch 7 of 100 cost=0.947425  time=(0.1s /1.6s)
    2019-03-22 14:25:23,016 [nnabla]: epoch 8 of 100 cost=0.949962  time=(0.1s /1.5s)
    2019-03-22 14:25:23,024 [nnabla]: epoch 9 of 100 cost=0.923598  time=(0.1s /1.4s)
    2019-03-22 14:25:23,045 [nnabla]: epoch 10 of 100 cost=0.898164  {train_error=0.903846, valid_error=0.788982} time=(0.1s /1.3s)
    2019-03-22 14:25:23,053 [nnabla]: epoch 11 of 100 cost=0.894671  time=(0.2s /1.4s)
    2019-03-22 14:25:23,064 [nnabla]: epoch 12 of 100 cost=0.881112  time=(0.2s /1.4s)
    ・・・
    2019-03-22 14:25:23,778 [nnabla]: epoch 88 of 100 cost=0.357768  time=(0.9s /1.0s)
    2019-03-22 14:25:23,789 [nnabla]: epoch 89 of 100 cost=0.317276  time=(0.9s /1.0s)
    2019-03-22 14:25:23,806 [nnabla]: epoch 90 of 100 cost=0.322348  {train_error=0.319766, valid_error=0.220777} time=(0.9s /1.0s)
    2019-03-22 14:25:23,815 [nnabla]: epoch 91 of 100 cost=0.315129  time=(0.9s /1.0s)
    2019-03-22 14:25:23,824 [nnabla]: epoch 92 of 100 cost=0.307717  time=(0.9s /1.0s)
    2019-03-22 14:25:23,832 [nnabla]: epoch 93 of 100 cost=0.334560  time=(0.9s /1.0s)
    2019-03-22 14:25:23,841 [nnabla]: epoch 94 of 100 cost=0.280173  time=(0.9s /1.0s)
    2019-03-22 14:25:23,850 [nnabla]: epoch 95 of 100 cost=0.291028  time=(1.0s /1.0s)
    2019-03-22 14:25:23,858 [nnabla]: epoch 96 of 100 cost=0.329578  time=(1.0s /1.0s)
    2019-03-22 14:25:23,867 [nnabla]: epoch 97 of 100 cost=0.295773  time=(1.0s /1.0s)
    2019-03-22 14:25:23,876 [nnabla]: epoch 98 of 100 cost=0.303806  time=(1.0s /1.0s)
    2019-03-22 14:25:23,885 [nnabla]: epoch 99 of 100 cost=0.303895  time=(1.0s /1.0s)
    2019-03-22 14:25:23,903 [nnabla]: epoch 100 of 100 cost=0.294800  {train_error=0.294852, valid_error=0.343269} time=(1.0s /1.0s)
    2019-03-22 14:25:23,903 [nnabla]: Training Completed.
    NNabla command line interface (Version 1.0.5.console_day3-fix-181220, Build 181219104847)
    2019-03-22 14:25:25,755 [worker]: [INFO]: create result_train.nnp
    2019-03-22 14:25:26,775 [worker]: [INFO]: create result.nnb
    2019-03-22 14:25:29,775 [worker]: [INFO]: create result.onnx
    2019-03-22 14:25:31,967 [worker]: [INFO]: worker done
    ```
* またこれまでと同様、学習カーブとして、以下のグラフが表示される。
    * cost
    * Training Error(学習誤差曲線)
    * validation Error(評価誤差曲線)
* 次に作成したモデルの評価を実行する。
* TRAININGタブ右側のRunボタンを選択することでモデルの評価が開始される。
* EVALUATION画面に遷移し、モデルの評価が始まる。
* 以下のようなログが出力され、評価が終了する。
    ```
    2019-03-22 14:26:50,821 [worker]: [INFO]: download: 13954/configurations/45896/data.sdcproj to work/data.sdcproj
    2019-03-22 14:26:51,368 [worker]: [INFO]: Find learned parameter file: results_current_100.nnp
    2019-03-22 14:26:52,460 [worker]: [INFO]: download: 13954/results/45896/result.nnp to results/result.nnp
    2019-03-22 14:26:53,035 [worker]: [INFO]: download: 13954/results/45896/results_best_90.nnp to results/results_best_90.nnp
    2019-03-22 14:26:53,615 [worker]: [INFO]: download: 13954/results/45896/results_current_100.nnp to results/results_current_100.nnp
    2019-03-22 14:26:53,619 [worker]: [INFO]: Use config file: 13954/results/45896/results_current_100.nnp, instead of config file: 13954/configurations/45896/data.sdcproj
    2019-03-22 14:26:53,652 [worker]: [INFO]: nnabla_cli forward -c /home/nnabla/results/results_current_100.nnp -d /dataset-cache/ccbf15a0-bcb6-4ba6-b10e-27fc877c4348/8.cache -o /home/nnabla/results
    2019-03-22 14:26:54,554 [nnabla]: data 16 / 30
    2019-03-22 14:26:54,555 [nnabla]: data 30 / 30
    2019-03-22 14:26:54,558 [nnabla]: Add output_result.zip to result.nnp.
    2019-03-22 14:26:54,558 [nnabla]: Add output_result.zip to results_best_90.nnp.
    2019-03-22 14:26:54,558 [nnabla]: Add output_result.zip to results_current_100.nnp.
    2019-03-22 14:26:54,559 [nnabla]: Forward Completed.
    NNabla command line interface (Version 1.0.5.console_day3-fix-181220, Build 181219104847)
    2019-03-22 14:26:55,241 [worker]: [INFO]: upload: results/output_result.csv to 13954/results/45896/output_result.csv
    2019-03-22 14:26:55,246 [worker]: [INFO]: create confusion_matrix.json
    2019-03-22 14:26:55,296 [worker]: [INFO]: confusion_matrix.json created
    2019-03-22 14:26:55,791 [worker]: [INFO]: upload: work/confusion_matrix.json to 13954/results/45896/confusion_matrix.json
    2019-03-22 14:26:56,337 [worker]: [INFO]: upload: results/result.nnp to 13954/results/45896/result.nnp
    2019-03-22 14:26:56,860 [worker]: [INFO]: upload: results/results_best_90.nnp to 13954/results/45896/results_best_90.nnp
    2019-03-22 14:26:57,380 [worker]: [INFO]: upload: results/results_current_100.nnp to 13954/results/45896/results_current_100.nnp
    2019-03-22 14:26:57,384 [worker]: [INFO]: create result_evaluate.nnp
    2019-03-22 14:27:01,515 [worker]: [INFO]: Completed to pageing file creator.
    2019-03-22 14:27:01,520 [worker]: [INFO]: worker done
    ```
* これまでと同様、評価結果として以下を選択して表示することができる。
    * Output Result
    * Confusion Matrix
* Output ResultにはIndex、x_0～x_3、y、ｙ'_0～y'_2が表示される。
* ここで先頭の評価データを確認してみる。先頭のデータ(=評価データのデータ番号が1)は0のアヤメであるため、yの値は0になっている。
* 評価により、このデータが0, 1, 2のどのアヤメ課の確率がそれぞれy'_0、y'_1、y'_2として格納されている。
* y'_0=0.9525...、y'_1=0.04687...、y'_2=0.00054...のため、y'_0が最も大きく、正しく分類ができていると言える。
* Confusion Matrixでは評価データを全て正しく分類できていることが分かる。
* ただし、実際は過学習していないかをきちんと確認しておく必要がある。
* 前述の手書き数字画像分類とアヤメの分類タスクを比較すると、誤差の収束が遅いことが分かる。
* ニューラルネットワークは手書き文字認識タスクの画像のような非構造化データに対して精度の高いモデルを生成するのに長けている一方、アヤメの分類問題のような構造化データに対してはアルゴリズムの良さを発揮できないこともある。


# ネットワーク構造の最適化
* 手書き文字分類タスクを例にニューラルネットワークを作成し、ネットワーク構造を最適化して分類精度を高めることを考える。
* PROJECT画面で以前に作成した"tutorial.basics.01_logistic_regression-trial"を選択する。
* EDIT画面で作成済みネットワークが表示されたら、右上の「Save as」ボタンでネットワークを別名プロジェクトとして保存する。
* ここでは"tutorial.basics.01_logistic_regression-trial4"とする。
* "tutorial.basics.01_logistic_regression-trial"ではMNISTの一部からデータセットを抽出し、Small MNISTとし、4と9の手書き文字を分類する2値分類を行なった。
* 以下では元のMNISTデータセットを扱うことにする。元のデータセットには手書き数字の0～9が含まれるため、タスクとしては多値分類となる。
* 右上のDATASETタブを選択すると、Training/Validationには以下のデータセットが選択されている。
    * Training：mnist.small_mnist_4or9_training
    * Validation：mnist.small_mnist_4or9_test
* Link Datasetのリンクを選択し、それぞれ以下のデータセットを選択する。
    * Training：mnist.mnist_training
    * Validation：mnist.mnist_test
* 次にEDIT画面に戻り、作成済みのネットワークを変更する。
* 元のネットワークの構成は以下の通りである。
    * 入力層：784ノード(28×28×1)
    * 出力層：1ノード(Affine), 活性化関数:Sigmoid
    * 誤差関数：2値分類交差エントロピー(Binary Cross Entropy)
* このネットワークを以下のように変更する。
    * 入力層：784ノード(28×28×1)
    * 中間層：128ノード(Affine), 活性化関数:ReLU 
    * 出力層：10ノード(Affine), 活性化関数:Softmax
    * 誤差関数：多値交差エントロピー(Categorical Cross Entropy)
* 各レイヤを配置・結線したら、各レイヤのパラメータを以下の通り変更する。パラメータの変更はレイヤを選択し、左下のLayer Propertyで行う。
    * 中間層のAffine2のOutshapeを100→128に変更(ReLU/DropoutのInput/Outputも自動的に128に変更される)
    * 出力層のAffineのOutshapeを1→10に変更(SoftmaxのInput/Outputも自動で10に変更される)
* 次にCONFIGタブを選択し、学習条件の設定を行なう。
* a
* ★★～P.110★★


# NNC操作方法Tips
* 足りないLayerの追加：
    * 左側のComponentsの追加可能Layer一覧からドラッグ&ドロップし、Mainタブ中に配置。
* Layer間の結線：
    * Layer上下の黒丸同士をマウスで接続
* Layer間の結線解除：
    * 結線を選択し、右上のActionからDelete
    * (結線を選択し、右クリック→Delete)
* 複数のLayerをまとめて操作：
    * ctrl押しながらLayerを選択

