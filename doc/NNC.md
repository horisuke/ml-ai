# Sample Project実行
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
* 画面上を右クリックすることでSave CSV as...のコンテキストが表示され、CSV形式で評価結果を保存することが可能。
* ★★～P.93★★