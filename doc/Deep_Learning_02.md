# 自然言語処理
* 日本語や英語など普段使っている言葉を自然言語(Natural Language)と言い、自然言語処理(NLP：Natural Language Processing)は自然言語を処理する分野、つまりコンピュータに人間の言葉を理解させるための技術のことである。
* 自然言語処理が扱う分野は多岐にわたるが、その本質はコンピュータに人間の言葉を理解させることにある。
* コンピュータが理解できる言語と言えば、プログラミング言語やマークアップ言語がある。
* それらの言語はコードの意味が一意に解釈できるように文法が定義されており、コンピュータはそのルールに従ってコードを解析する。
* 一般的なプログラミング言語は機械的で無機質であり、"固い言語"である。
* 一方、自然言語は同じ意味の文章でも様々な表現が可能であり、曖昧さがあったり、柔軟に意味や形が変わることから"柔らかい言語"であると言える。
* このような柔らかい言語である自然言語をコンピュータに理解させるのは難しい。
* 一方、その問題をクリアできれば、人にとって役立つことをコンピュータに行わせることができると言える。
* 実際にそのような例は多くあり、検索エンジンや機械翻訳、質問応答システム、文章の自動翻訳や感情分析などが挙げられる。
* 自然言語は文字によって構成され、また自然言語の言葉の意味は単語によって構成される。
* 単語は意味の最小単位であり、自然言語をコンピュータに理解させるためには、単語の意味を理解させることが重要であると言える。
* コンピュータに単語の意味を理解させるために、単語の意味をうまく捉えた以下の表現方法について考える。
    * シソーラスによる手法
    * カウントベースによる手法
    * 推論ベースの手法(word2vec)


# シソーラス
* 単語の意味を表す方法として、辞書のように1つ1つの単語に対してその意味を説明する方法が考えられる。
* このように単語の意味を定義することでコンピュータで単語の意味を理解する。
* ただし、これまでの自然言語処理の歴史では、人が使う辞書とは異なるシソーラス(thesaurus)と呼ばれるタイプの辞書が使われてきた。
* シソーラスは基本的には類語辞書であり、同じ意味の単語(同義語)、意味の似た単語(類義語)が同じグループに分類されている。
* 例えば、シソーラスを使うことでcarの同義語にはautomobileやmotocarなどが存在することがわかる。
* また、シソーラスでは、「上位と下位」、「全体と部分」などのより細かい関係性が単語の間で定義されている場合がある。
* 例えば、グラフ構造によって、各単語の関係性が定義されているケースがある。
* carという単語の上位概念としてmotor vehicleがあり、下位概念としてSUVやcompactがある。
* このように全ての単語に対して類義語の集合を作り、それぞれの単語の関係をグラフで表現することで、単語間の繋がりを定義することができる。
* この構造を使用することでコンピュータに単語間の関連性を教えることができる。


# 代表的なシソーラス - WordNet
* 自然言語処理分野におて、有名なシソーラスとして、WirdNetがある。
* WordNetはプリンストン大学で1985年に開発がスタートしており、これまで多くの研究に利用されてきた。
* WordNetを使うことで類義語を取得できたり、単語ネットワークを利用したりすることができる。
* さらに単語ネットワークを利用することで単語間の類似度を算出することも可能。


# シソーラスの問題点
* シソーラスにより、多くの単語に対して、同義語や階層構造の関係性を定義することによってコンピュータに単語の意味を教えることができる。
* 一方、それらを人の手でラベル付けすることには以下のような大きな欠点が存在する。
    * 時代変化に伴う対応が困難
        * 自然言語は生きており、絶えず新しい言葉が生まれ、古い言葉は忘れ去られる。
        * また時代変化に伴い、言葉の意味が変化する場合もある。
        * そのような単語の変化に対応するためにはシソーラスを人手で絶えず更新する必要がある。
    * 人手によるラベル付けのコストが高い
        * シソーラスを作るためには多くのコストがかかる。
        * 現存する英単語の総数は1000万語を超えると言われているが、それら全てに単語の関連づけをする必要がある。
    * 単語の細かいニュアンスを表現できない
        * シソーラスでは類義語として似た単語をグループ化するが、実際には似た単語であっても、それぞれニュアンスが少し異なる。
        * 単語ごとの微妙なニュアンスの差異はシソーラスでは表すことができない。
* 上記の通り、シソーラスを使う手法(単語の意味を人手によって定義する方法)には大きな問題がある。
* この問題を避けるため「カウントベースの手法」、ニューラルネットワークを使った「推論ベースの手法」がよく用いられる。
* これらの手法では、大量のテキストデータから自動的に単語の意味を抽出する。


# カウントベースの手法
* カウントベースの手法ではコーパス(corpus)を使用する。
* コーパスとは大量のテキストデータであるが、やみくもに集められたテキストデータではなく、自然言語処理の研究やアプリケーション開発のために目的をもって収集されたテキストデータのことを呼ぶ.
* コーパスはテキストデータに過ぎず、含まれる文章は人の手によって書かれたものである。
* これはコーパスに自然言語に対する人間の知識(文章の書き方、単語の選び方、単語の意味)が含まれていることを意味する。
* カウントベースの手法では人間の知識が詰まったコーパスから自動的かつ効率的にそのエッセンスを抽出することを行う。
* 自然言語処理分野で用いられるコーパスにはテキストデータに対してさらに追加の情報が付与されていることがある。
* 例えば、テキストデータの個々の単語に対して、品詞がラベル付けされているケースなどがある。
* その場合、コンピュータが扱いやすいようにコーパスが機構造などのデータ形式として構造化されていることが一般的である。
* 自然言語処理分野で用いられるコーパスには様々なものが存在し、WikipediaやGoogle Newsなどのテキストデータや、シェクスピアや夏目漱石などの偉大な作家の作品群がコーパスとして有名である。
* ここではまずはじめに1文からなる単純な以下のようなテキストをコーパスとして利用する。
    ```python
    >>> text = 'You say goodbye and I say hello.'
    ```
* 次にPythonの対話モードで小さなテキストデータに前処理(テキストデータを単語に分割し、その分割した単語を単語IDのリストへ変換)を行う。
    ```python
    >>> text = text.lower()
    >>> text = text.replace('.', '. ')
    >>> text
    'you say goodbye and i say hello .'

    >>> words = text,split(' ')
    >>> words
    ['you', 'say', 'goodbye', 'and', 'i', 'say', 'hello', '.']
    ```
* 上記では、まずlower()で全て小文字に変換し、ピリオドの前にスペースを挿入する。
    * 実際はより賢く汎用的なやり方として正規表現を使用する方法がある。
    * reモジュールをimportし、re.split('(\W+)?', text)とすることで単語単位に分割できる。
* 次にsplit()でスペースを区切り文字として、単語に分割し、wordsというリストに各単語を要素として保持する。
* これで元の文章を単語のリストとして使用できるようになる。
* ただ単語をテキストのまま操作するのは不便なため、単語にIDを振り、IDのリストとして使用できるようにする。
* まず以下の通り、単語のIDと単語の対応表をPythonのディクショナリで作成する。
    ```python
    >>> word_to_id = {}
    >>> id_to_word = {}
    >>>
    >>> for word in words:
            if word not in word_to_id:
                new_id = len(word_to_id)
                word_to_id[word] = new_id
                id_to_word[new_id] = word
    ```
* id_to_wordにはキーを単語ID、値を単語として保持し、word_to_idにはキーを単語、値を単語IDとして保持する。
* if文では単語がword_to_idにまだ保持されていない場合にword_to_idの要素数をnew_idとして払い出し、word_to_idにそのnew_idを値として追加する。
* id_to_wordにはそのnew_idをキーとしてwordを追加する。
* 上述のtextの文はid_to_word、word_to_idにそれぞれ以下のように保持される。
    ```python
    >>> id_to_word
    {0:'you', 1:'say', 2:'goodbye', 3:'and', 4:'i', 5:'hello', 6:'.'}
    >>> word_to_id
    {'you':0, 'say':1, 'goodbye':2, 'and':3, 'i':4, 'hello':5, '.':6}
    ```
* これらのディクショナリを使うことにより、単語から単語IDを検索したり、単語IDから単語を検索したりできる。
    ```python
    >>> id_to_word[1]
    'say'
    >>> word_to_id['hello ']
    5
    ```
* 最後に以下の通り、単語のリストwordsを単語IDのリストcorpusに変換し、その結果をNumpy配列にして保持する。
    ```python
    >>> import numpy as np
    >>> corpus = [word_to_id[w] for w in words]
    >>> corpus = np.array(corpus)
    >>> corpus
    array([0,1, 2, 3, 4, 1, 5, 6])
    ```
* for文の内包表記ではwordsの先頭から要素の値を順に取り出してwに格納し、そのwをキーとしてword_to_idの値、つまり単語IDを取得する。
* 取得した単語idはcorpusにリストとして保持される。
* 次の処理でcorpusはnp.array()でNumpy配列に変換される。
* ここまでの処理を前処理として、preprocess()という関数にしてまとめて実装すると、以下のようになる。
    ```python
    def preprocess(text):
        text = text.lower()
        text = text.replace('.', '. ')
        words = text,split(' ')

        word_to_id = {}
        id_to_word = {}
        for word in words:
            if word not in word_to_id:
                new_id = len(word_to_id)
                word_to_id[word] = new_id
                id_to_word[new_id] = word

        corpus = np.array([word_to_id[w] for w in words])

        return corpus, word_to_id, id_to_word
    ```
* この関数を使用すると、コーパスの前処理は以下のようになる。
    ```python
    >>> text = 'You say goodbye and I say hello.'
    >>> corpus, word_to_id, id_to_word = preprocess(text)
    ```
* ここまででコーパスを扱う準備が整ったので、これを用い、単語の意味を抽出する。
* ここでカウントベースの手法を用い、単語をベクトルで表すことを行なう。
* ベクトル表現の例として「色」を考える。
* 色には様々な固有の名前が付けられている一方、RGBの3成分がその色にどれだけ含まれるかによって色を表現することもできる。
* 前者は色の数だけ異なる名前を命名することになるが、後者は色を3次元ベクトルで表すことになる。
* ここで、RGBによる色のベクトル表現の方が正確に色を指定できると言える。
* また例えば、(R, G, B) = (201, 23, 30)であれば、その色の名前を知るよりも、赤系の色だとイメージがしやすいと言える。
* 加えて、色同士の関連性・類似性もベクトル表現の方が容易に判断でき、また定量化も容易にできる。
* このような色のベクトル表現を単語でも行えないかを考える。
* つまり、コンパクトかつ理にかなったベクトル表現を「単語」というフィールドにおいて構築することを考える。
* これが構築できれば、「単語の意味」を的確に捉えたベクトル表現が可能となる。
* このベクトル表現を単語の分散表現と呼ぶ。
* 単語の分散表現では単語を固定長のベクトルで表現し、そのベクトルは密なベクトル(ベクトルの各要素が0でない実数で構成されるベクトル)で表現される。
* これまでの自然言語処理の研究で行われてきた単語をベクトルで表現する手法は「単語の意味は周囲の言語によって形成される」というアイデアに基づいているものがほとんどである。
* これは分布仮説(distributional hypothesis)と呼ばれ、単語自体に意味はなく、その単語のコンテキスト(文脈)によって、単語の意味が形成される、というものである。
* 意味的に同じ単語は同じような文脈で多く出現する。
    * 例)"I drink beer."、"We drink wine."などの文章におけるdrink
* また、以下のような文章があれば、"guzzle"が"drink"と同じような文脈で使われること、意味が近い単語である、ということもわかる。
    * "I guzzle beer."、"We guzzle wine."
* 注目する単語に対するその周囲に存在する単語をコンテキストと呼び、コンテキストに周囲のどこまでの単語を含めるかのサイズをウィンドウサイズ(Window size)と呼ぶ。
* つまりウィンドウサイズが2の場合、左右のそれぞれ2単語がコンテキストとなる。
* ウィンドウサイズは左右均等である必要はなく、どちらか一方だけをコンテキストとする、といったことや分の区切りを考慮したコンテキストを考える、といったこともできる。
* 上記をもとに分布仮説に基づいて単語をベクトルで表現する方法を考える。
* もっとも単純な方法はある単語の周囲にどのような単語がどの程度出現するかをカウントし、それを集計することである。
* まず前述のpreprocess()関数を使って下準備を行う。
    ```python
    import sys
    sys.path.append('..')
    import numpy as np
    from common.util import preprocess

    text = 'You say goodbye and I say hello.'
    corpus, word_to_id, id_to_word = preprocess(text)

    print(corpus)
    # [0,1, 2, 3, 4, 1, 5, 6]

    print(id_to_word)
    # {0:'you', 1:'say', 2:'goodbye', 3:'and', 4:'i', 5:'hello', 6:'.'}
    ```
* 上記より、textは単語数は8、語彙数は7であることがわかる。
* この各単語について、ウィンドウサイズを1とし、コンテキストに含まれる単語の頻度を数えると以下のようになる。
    |         | you   | say   | goodbye | and   | i     | hello | .     |
    |:-------:|:-----:|:-----:|:-------:|:-----:|:-----:|:-----:|:-----:|
    | you     | 0     | 1     | 0       | 0     | 0     | 0     | 0     |
    | say     | 1     | 0     | 1       | 0     | 1     | 1     | 0     |
    | goodbye | 0     | 1     | 0       | 1     | 0     | 0     | 0     |
    | and     | 0     | 0     | 1       | 0     | 1     | 0     | 0     |
    | i       | 0     | 1     | 0       | 1     | 0     | 0     | 0     |
    | hello   | 0     | 1     | 0       | 0     | 0     | 0     | 1     |
    | .       | 0     | 0     | 0       | 0     | 0     | 1     | 0     |
* 1行目の'you'は隣り合っている'say'のみがコンテキストとなるため、1となり、それ以外の単語は0となる。
* これは同時に'you'という単語が[0, 1, 0, 0, 0, 0, 0]というベクトルで表現できることを意味している。
* これを全ての語彙に対して行ない作成されるテーブルを共起行列(co-occurence matrix)と呼ぶ。
    * 共起とは、ある単語がある文中に出現したとき、その文中に別の単語が頻繁に出現すること。
* この共起行列をnp.arrayで定義すると以下のようになる。
    ```python
    C = np.array([
        [0, 1, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 1, 1, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 1, 0]
    ], dtype=np.int32)
    ```
* この共起行列を使うことで以下の通り、各単語のベクトルを取得することができる。
    ```
    print(C[0]) # 単語IDが0のベクトル
    # [0, 1, 0, 0, 0, 0, 0]

    print(C[4])  # 単語IDが4のベクトル
    # [0, 1, 0, 1, 0, 0, 0]

    print(C[word_to_id['goodbye']]) # 'goodbyeのベクトル'
    # [0, 1, 0, 1, 0, 0, 0]
    ```
* これを踏まえ、コーパスから共起行列を作る関数create_co_matrix()を実装する。
    * create_co_matrix(corpus, vocab_size, window_size=1)
        * corpus：単語IDのリスト
        * vocab_size：語彙数
        * window_size：ウィンドウサイズ
    ```python
    def create_co_matrix(corpus, vocab_size, window_size=1):
        corpus_size = len(corpus)
        co_matrix = np.zeors((vocab_size, vocab_size), dtype=np.int32)

        for idx, word_id in enumerate(corpus):
            for i in range(1, window_size+1):
                left_idx = idx - i
                right_idx = idx + i

                if left_idx >= 0:
                    left_word_id = corpus[left_idx]
                    co_matrix[word_id, left_word_id] += 1
                
                if right_idx < corpus_size:
                    right_word_id = corpus[right_idx]
                    co_matrix[word_id, right_word_id] += 1

        return co_matrix
    ```
* まず語彙数×語彙数の大きさの2次元配列を全要素を0で初期化そて生成する。
* 次にコーパス中の各単語に対して、そのウィンドウに含まれる単語をカウントしていく。
* ただし、コーパスの右端・左端のはみ出しチェックを行い、はみ出していない単語のみカウントする。
* この関数を用いることで任意の大きさのコーパスに対する共起行列を生成することができる。
* 次にベクトル間の類似度を計測する方法を考える。
* 一般的には様々な方法があり、ベクトルの内積やユークリッド距離などが代表的な方法として挙げられる。
* 単語のベクトル表現の類似度に関しては、コサイン類似度(cosine similarity)がよく用いられる。
* コサイン類似度はベクトルx, yに対して、以下の(2.1)で定義される。
    * similarity(x, y) = (x・y) / (||x| ||y||) = (x1y1 + x2y2 +・・・xnyn) / (√x1^2+・・・+xn^2 √y1^2+・・・+yn^2)　(2.1)
        * x = (x1, x2, ... , xn)
        * y = (y1, y2, ... , yn)
* (2.1)は分子にベクトルの内積、分母に各ベクトルのノルムがある。
* ノルムはベクトルの大きさを表したのもので、ここではL2ノルム(ベクトルの各要素の2乗和の平方根)を用いている。
* これはベクトルを正規化して内積を取っていると言える。
* コサイン類似度は直感的には2つのベクトルがどれだけ同じ方向を向いているかを表すものである。
* 2つのベクトルが完全に同じ方向を向いている場合、コサイン類似度は1となり、完全に逆向きだと-1となる。
* これにより、コサイン類似度を実装すると以下のようになる。
    ```python
    def cos_similarity(x, y):
        nx = x / np.sqrt(np.sum(x**2))　＃xの正規化
        ny = y / np.sqrt(np.sum(y**2))　＃yの正規化

        return np.dot(nx, ny)
    ```
* 引数x, yはNumPy配列とし、x, yをそれぞれ正規化した後、両者の内積を求めている。
* 一方、上記の実装だとx, ｙに0ベクトルが渡されてくると、0除算が発生してしまう。
* ここでは、小さな値epsを1e-8とし、引数で指定できるようする。
* 指定されたepsはnx, nyの分母の計算に必ず加算される値として実装する。
* これを踏まえたcos_similarity()の改良版は以下の通りとなる。
    ```python
    def cos_similarity(x, y, eps=1e-8):
        nx = x / np.sqrt(np.sum(x**2)) + eps)　＃xの正規化
        ny = y / np.sqrt(np.sum(y**2)) + eps)　＃yの正規化

        return np.dot(nx, ny)
    ```
* このcos_similarity()改良版を用いて、"you"と"i"の単語ベクトルの類似度を求める実装は以下の通りになる。
    ```python
    import sys
    sys.path.append('..')
    from common.util import preprocess, create_co_matrix, cos_similarity

    text = 'You say goodbye and I say hello.'
    corpus, word_to_id, id_to_word = preprocess(text)
    vocab_size = len(word_to_id)
    C = create_co_matrix(corpus, vocab_size)

    c0 = C[wprd_to_id['you']] # 'you'の単語ベクトル
    c1 = C[wprd_to_id['i']]   # 'i'の単語ベクトル
    print(cos_similarity(c0, c1))
    # 0.7071067691154799
    ```
* "you"と"i"の単語ベクトルの類似度は0.70...と言える。
* コサイン類似度は1から-1までの値を取るので、この値は比較的高い値(=類似正がある)と言える。
* 次にある単語がクエリとして与えられたとき、そのクエリに対して類似した単語を上位から順に表示する関数most_similar()を考える。
    * most_similar()
        * query：クエリ(単語)
        * word_to_id：単語から単語IDへのディクショナリ
        * id_to_word：単語IDから単語へのディクショナリ
        * word_matrix：単語ベクトルをまとめた行列で各行に対応する単語のベクトルが格納(共起行列)
        * top：取得する類似度のランキング
* 実装は以下の通り。
    ```python
    def most_similar(query, word_to_id, id_to_word, word_matrix, top=5):
        # Pick up the query
        if query not in word_to_id:
            print('%s is not found' % query)
            return

        print('\n[query] ' + query)
        query_id = word_to_id[query]
        query_vec = word_matrix[quiery_id]

        # calculate cosine similarity
        vocab_size = len(id_to_word)
        similarity = np.zeros(vocab_size)
        for i in range(vocab_size):
            similarity[i] = cos_similarity(word_matrix[i], query_vec)
        
        # Output the value of cosine similarity in descending order 
        count = 0
        for i in (-1 * similarity).argsort():
            if id_to_word[i] == query:
                continue
            print('%s: %s' % (id_to_word[i], similarity[i]))

            count +=1
            if count >= top:
                return
    ```
* まずはじめに渡されてきたqueryがword_to_idのキーに存在するかをチェックする
* 存在する場合は処理をそのqueryを表示し、処理を続行する。
* word_to_idからqueryをキーにその単語のIDを取得する。取得した単語IDを使い、word_matrixからその単語の単語ベクトルを取得し、query_vecに保持する。
* 次にコサイン類似度similarityを計算する。まず、id_to_wordの大きさ分のベクトルを要素0で初期化する。
* cos_similarity()にword_matrixの各行とquery_vecを渡し、コサイン類似度を計算し、各行の値をベクトルsimilarityの要素として保持する。
* 最後にコサイン類似度をその値が高い順に表示するためにargsort()関数を使用する。
* argsort()はNumpy配列の要素を小さい順にソートし、ソート後のインデックスを返す関数である。
    ```python
    >>> x = np.array([100, -20, 2])
    >>> x.argsort()
    # array([1, 2, 0])
    ```
* よって、similarityの各要素に-1をかけ、argsort()関数を適用すると、その返り値は値の大きい要素順のインデックスが変えることになる。
* これによって、for文を回し、id_to_word, similarityを表示する。
* このfor文をtopで渡された回数分回すことで、
* コサイン類似度が高い順にその値と単語を表示することができる。
* 次にこのmost_similar()関数を使用する例は以下の通り。例ではクエリとして'you'を指定する。
    ```python
    import sys
    sys.path.append('..')
    from common.util import preprocess, create_co_matrix, most_similar

    text = 'You say goodbye and I say hello.'
    corpus, word_to_id, id_to_word = preprocess(text)
    vocab_size = len(word_to_id)
    C = create_co_matrix(corpus, vocab_size)

    most_similar('you', word_to_id, id_to_word, C, top=5)
    ```
* これを実行すると以下のような結果が得られる。
* この結果から'you'に近い単語は3つあり、'goodbye', 'i', 'hello'であることがわかる。
    ```
    [query] you
    goodbye: 0.7071067691154799
    i: 0.7071067691154799
    hello: 0.7071067691154799
    say: 0.0
    and: 0.0
    ```
* 'i'は'you'と同じ人称代名詞であるため、コサイン類似度が高いことはよく理解できる。
* 一方、'goodbye'や'hello'のコサイン類似度が高いには説明がつかないが、これはコーパスのサイズが極端に短いことが原因と考えられる。


# カウントベースの手法の改善
* 前述の通り、単語の共起行列を作り、単語をベクトルで表現することができたが、さらに共起行列を改善する。
* 前述の共起行列の要素は2つの単語が共起した回数を表している。
* ただ、これは特に高頻度単語に目を向けると、あまり良い性質を持たないことがわかる。
* 例えば、コーパスにおいて、「the」、「car」の共起を考える。
* その場合、コーパスでは「...the car...」というフレーズは多くみられることが考えられ、共起回数は大きくなる。
* 一方、「car」と「drive」は単語間に明らかに強い関連性があるにもかかわらず、出現回数だけを見ると、「car」は「drive」よりも「the」の方が強い関連性を持ってしまうことになる。
* このような問題を解決するため、相互情報量(PMI：Pointwise Mutual Information)と呼ばれる指標が使われる。
* PMIは確率変数x, yに対して次の式で定義され、値が大きいほどx, yの関連性が高いと言える。
    * PMI(x, y) = log_2 ((P(x, y)) / (P(x)P(y)))　　(2.2)
        * P(x)：xが起きる確率(単語xがコーパスに出現する確率)
        * P(y)：yが起きる確率(単語yがコーパスに出現する確率)
        * P(x, y)：xとyが同時に起きる確率(単語x, yが共起する確率)
* 例えば、10000個の単語からなるコーパスで「the」が100回出現するとする。P('the') = 100/10000 = 0.01となる。
* 次にy = 'car'とし、「the」と「car」が10回共起したとすると、P('the', 'car') = 10/10000 = 0.001となる。
* 共起行列(共起した単語の回数を要素に持つ行列)をC、単語x, yの出現回数及び単語x,yの共起回数をそれぞれC(x), C(y), C(x,y)とし、コーパスに含まれる単語数をNとすると、(2.2)は以下のように表すことができる。
    * PMI(x, y) = log_2 ((P(x, y)) / (P(x)P(y))) = log_2 ((C(x,y)/N) / (C(x)/N・C(y)/N)) = log_2 ((C(x,y)・N) / (C(x)C(y))　　(2.3)
* (2.3)が導出できたことにより、PMIは共起行列から求めることが可能であることがわかる。
* この式を用い、具体的なコーパスに関してPMIを算出する。
* N=10000のコーパスにおいて、「the」「car」「drive」がそれぞれ1000回、20回、10回出現したとする。
* また、「the」と「car」、「car」と「drive」の共起はそれぞれ10回、5回とする。
* この時点で従来の共起行列の観点では、「the」と「car」の方が「car」と「drive」より関連性が高いことが言える。
* 一方、「the」と「car」、「car」と「drive」のPMIを算出すると、以下のようになる。
    * PMI('the', 'car') = log_2 (10・10000)/(1000・20) ≒ 2.32
    * PMI('car', 'drive') = log_2 (5・10000)/(20・10) ≒ 7.97
* 上記より、PMIの観点では、「car」と「drive」の方が「the」と「car」より関連性が高いという結果になる
* これは上述の通り、PMIにより単語単独の出現回数が考慮されてことによると言える。
* 一方、PMIは2つの単語の共起回数が0になる場合、log_2 0 = ∞となってしまう。
* よって、実際には以下で示す、正の相互情報量(PPMI：Positive PMI)が使われる。
    * PPMI(x, y) = max(0, PMI(x,y))　　(2.6)
* これにより、PMIの値がマイナスの場合は、それを0として扱うことができ、単語間の関連度を0以上の実数で表すことができる。
* 共起行列をPPMI行列(要素がPPMI値の行列)に変換する関数を実装すると、以下の通りとなる。
    ```python
    def ppmi(C, verbose=False, eps=1e-8):
        M = np.zeros_like(C, dtype=np.floar32)
        N = np.sum(C)
        S = np.sum(C, axis=0)
        total = C.shape[0] * C.shape[1]
        cnt = 0

        for i in range(C.shape[0]):
            for j in range(C.shape[1]):
                pmi = np.log2(C(i, j) * N / (S[j]*S[i]) + eps)
                M[i, j] = max(0, pmi)

                if verbose:
                    cnt+=1
                    if cnt % (total//100) == 0:
                        print('%lf%% done', %(100*cnt/total))
        
        return M
    ```
* この関数を用いて、共起行列をPPMI行列に変換する実装は以下のようになる。
    ```python
    import sys
    sys.path.append('..')
    import numpy as np
    from common.util import preprocess, create_co_matrix, cos_similarity, ppmi

    text = 'You say goodbye and I say hello.'
    corpus, word_to_id, id_to_word = preprocess(text)
    vocab_size = len(word_to_id)
    C = create_co_matrix(corpus, vocab_size)
    W = ppmi(C)

    np.set_printoptions(precision=3)
    print('covariance matrix')
    print(C)
    print('-'*50)
    print('PPMI')
    print(W)
    ```
* 上記を実行すると以下の結果が得られる。
    ```
    covarience matrix
    [[0 1 0 0 0 0 0]
     [1 0 1 0 1 1 0]
     [0 1 0 1 0 0 0]
     [0 0 1 0 1 0 0]
     [0 1 0 1 0 0 0]
     [0 1 0 0 0 0 1]
     [0 0 0 0 0 1 0]]
     --------------------------------------------------
     PPMI
    [[0.     1.807  0.     0.     0.     0.     0.   ]
     [1.807  0.     0.807  0.     0.807  0.807  0.   ]
     [0.     0.807  0.     1.807  0.     0.     0.   ]
     [0.     0.     1.807  0.     1.807  0.     0.   ]
     [0.     0.807  0.     1.807  0.     0.     0.   ]
     [0.     0.807  0.     0.     0.     0.     2.807]
     [0.     0.     0.     0.     0.     2.807  0.   ]]
    ```
* これにより、PPMI行列を取得することができ、共起行列より良い指標の単語ベクトルを取得できたと言える。
* ただ、PPMI行列はこーますの語彙数が増えるにつれ、各単語の単語ベクトルの次元数が増えていくという問題がある。
* つまり、コーパスに含まれる語彙数が10万個の場合、そのベクトルの次元数も同様に10万となることを意味する。
* PPMI行列はその要素の多くが0であり、これはほとんどの要素が重要ではなく、各要素の持つ重要度が低いことを意味する。
* このように0要素が多いベクトルはノイズに弱く、頑健性に乏しいという欠点がある。
* これらの問題に対してはベクトルの次元削減(dimensionality reduction)を行なう。
* ただし、単純に次元を削減するのではなく、重要な情報をできるだけ残したうえで削減する必要がある。
* ここでの次元削減ではデータの分布を見て、重要な軸を見つけることを行なう。
* 例えば、2次元の座標で表されているデータ点を1次元の座標で表すとする。
* その際はデータの広がりを考慮し、その広がりが大きな軸を見つけ、新たな軸を導入する。
* 新しい軸を見つけ、導入する場合、各データ点の値は新しい軸へ射影された値に変換される。
* データの広がりを考慮した軸が取れていれば、1次元の値に変換されても、データの本質的な差異は失わずに捉えたままにできる。
* ベクトル中のほとんどの要素が0である行列は「疎な行列」と呼ばれる。
* 疎な行列から重要な軸を見つけ、より少ない次元で表現し直すことで、次元が少なくほとんどの要素が0でない行列(密な行列)に変換することを考える。
* 次元削減にはいくつかの方法があり、ここでは特異値分解(SVD：Singular Value Decomposition)を用いる。
* SVDでは任意の行列Xを3つの行列U, S, Vの積へと分解する。式で書くと以下の通りとなる。
    * X = U・S・V^T　　(2.7)
* ここで、U及びVは直交行列(=両行列の積が単位行列)、Sは対角行列(対角成分以外の成分が全て0である行列)となる。
* 直交行列Uは単語空間の軸を形成しており、対角行列Sの対角成分には「特異値」が大きい順に並んでいる。
* この特異値は単語空間の軸の重要度とみなすことができる。
* つまり、行列Sの特異値が小さいものは重要度が低いとみなし、それらの要素を削ることで次元を削減し、元の行列の近似した行列を取得する。
* これは単語のPPMI行列から重要度の低い要素を削減した単語ベクトルU'を取得することを意味する。
* SVDの実装はNumpyのlinalgモジュールにあるsvdメソッドで実行できる。
    * linalgはlinear algebra(線形代数)を表す。
* SVDを適用する実装は以下の通り。
    ```python
    import sys
    sys.path.append('..')
    import numpy as np
    from common.util import preprocess, create_co_matrix, ppmi

    text = 'You say goodbye and I say hello.'
    corpus, word_to_id, id_to_word = preprocess(text)
    vocab_size = len(word_to_id)
    C = create_co_matrix(corpus, vocab_size, window_size=1)
    W = ppmi(C)

    U, S, V = np.linalg.svd(W)
    ```
* 上記にて得られる行列UにSVDによって変換された密な単語ベクトル表現が格納されている。
* 共起行列、PPMI行列、SVDで得られる行列を表示すると以下の通り。
    ```python
    print(C[0])　　# 共起行列
    # [0 1 0 0 0 0 0]

    print(W[0])　　# PPMI行列
    # [0.     1.807  0.     0.     0.     0.     0.   ]

    print(U[0])　　# SVD
    # [3.409e-01  -1.110e-16  -1.205e-01  -4.441e-16  0.000e+00  -9.323e-01  2.226e-16]
    ```
* 上記より、疎なベクトルW[0]がSVDによって密なベクトルU[0]に変換されていることがわかる。
* この密なベクトルを次元削減する場合、その先頭の要素を必要な分だけ取り出すことで実現できる。
* 例えば、2次元ベクトルに削減する場合は以下のようになる。
    ```python
    print(U[0, :2])
    # [3.409e-01  -1.110e-16]
    ```
* 次に各単語を2次元ベクトルで表すことにし、それを以下の実装でグラフにプロットしてみる。
    ```python
    for word_id in word_to_id.items():
        plt.annotate(word, (U[word_id, 0], U[word_id, 1]))
    
    plt.scatter(U[:,0], U[:,1], alpha=0.5)
    plt.show()
    ```
* plt.annotate(word, x, y)により、2次元グラフ上の座標(x,y)にwordというテキストを描画する。
* 結果を見ると、「goodbye」と「hello」、「you」と「i」が近い位置にいることがわかる。
* SVDの計算は行列のサイズをNとして場合、O(N^3)というオーダーの計算量になる。
* これは現実的に非常に大きな計算量となるため、通常はTruncated SVDのような高速化手法を用いるのが一般的である。
* Truncated SVDでは特異値が小さなものを切り捨てることで高速化を図っている。
* 上記の結果は、コーパスが小さく結果が正しく出ていない部分もある。
* 以下では、PTBデータセットというより大きなコーパスを使って同じ処理を行って結果を確認する。
* PTBはPenn Treebankと呼ばれるコーパスのことで、提案手法の品質を測定するためのベンチマークとしてよく利用されている。
* PTBコーパスはWord2vecの発明者であるTomas MikolovのWebページで用意されているものでテキストファイルでて供されているが、以下のような前処理が施されている。
    * レアな単語を\<unk>に置換
    * 具体的な数字をNに置換
* 以下ではこれら前処理が施されているテキストデータをPTBコーパスとして使用する。
* PTBコーパスは1つの文が1行ごとに保存されており、ここでは各文を連結したものを1つの大きな時系列データとして扱うことにする。
* 加えて各文の終わりに\<eos>という特殊文字(eos=end of sentence)を挿入する。
* Penn Treebankのデータセットを扱う実装は以下の通り。
    ```python
    import sys
    sys.path.append('..')
    from dataset import ptb

    corpus, word_to_id, id_to_word = ptb.load_data('train')

    print('corpus size:', len(corpus))
    print('corpus[:30]', corpus[:30])
    print()
    print('id_to_word[0]:', id_to_word[0])
    print('id_to_word[1]:', id_to_word[1])
    print('id_to_word[2]:', id_to_word[2])
    print()
    print('word_to_id['car']:', word_to_id['car'])
    print('word_to_id['happy']:', word_to_id['happy'])
    print('word_to_id['lexus']:', word_to_id['lexus'])
    ```
* 上記を実行すると、以下の結果が得られる。
    ```
    corpus size:929589
    corpus[:30]: [0 1 2 3 4 5 6 7 8...27 28 29]

    id_to_word[0]:aer
    id_to_word[1]:banknote
    id_to_word[2]:berlitz

    word_to_id['car']: 3856
    word_to_id['happy']: 4428
    word_to_id['lexus']: 7426
    ```
* corpusには単語IDのリストが格納され、id_to_word, word_to_idにはそれぞれ、単語IDから単語への変換ディクショナリ、単語から単語IDへの変換ディクショナリが格納される。
* ptb.load_data()ではPTBのデータをloadする。引数で'train', 'test', 'valid'を渡すことにより、それぞれ学習用、テスト用、検証用のデータが指定できる。
* PTBデータセットに対してカウントベースの手法を適用することを考える。
* ここではより高速なSVDを使用するため、sklearnをimportし、randomized_svd()メソッドを使用する。
* randomized_svd()では乱数を使ったTruncated SVDで特異値の大きなものだけに限定して計算し、通常のSVDよりも高速に計算を行うことが可能。
* 実装は以下の通り。
    ```python
    import sys
    sys.path.append('..')
    import numpy as np
    from common.util import most_similar, create_co_matrix, ppmi
    from dataset import ptb

    window_size = 2
    wordvec_size = 100

    corpus, word_to_id, id_to_word = ptb.load_data('train')
    vocab_size = len(word_to_id)
    print('counting co-occurrence...')
    C = create_co_matrix(corpus, vocab_size, window_size=1)
    print('calculating PPMI...')
    W = ppmi(C, verbose=True)

    print('calculating SVD...')
    try:
        # truncated SVD
        from sklearn.utils.extmath import randomized_svd
        U, S, V = randomized_svd(W, n_components=wordvec_size, n_iter=5, random_state=None)
    except ImportError:
        # SVD
        U, S, V = np.linalg.svd(W)
    
    word_vecs = U[:, :wordvec_size]

    querys = ['you', 'year', 'car', 'toyota']
    for query in querys:
        most_similar(query, word_to_id, id_to_word, word_vecs, top=5)
    ```
* 上記を実行すると、以下の結果が得られる。
    ```
    [query] you
     i: 0.702039909619
     we: 0.699448543998
     've: 0.554828709147
     do: 0.534370693098
     else: 0.512044146526

    [query] year
     month: 0.731561990308
     quarter: 0.658233992457
     last: 0.622425716735
     earlier: 0.607752074689
     next: 0.601592506413
    
    [query] car
     luxury: 0.620933665528
     auto: 0.615559874277
     cars: 0.569818364381
     vehicle: 0.498166879744
     corsica: 0.472616831915
    
    [query] toyota
     motor: 0.738666107068
     nissan: 0.677577542584
     motors: 0.647163210589
     honda: 0.628862370943
     lexus: 0.604740429865
    ```
* 上記の結果を見てみると、まず「you」というクエリに対して、「i」、「we」といった人称代名詞が上位を占めており、文法的に共通の単語が該当している。
* 「year」がクエリの場合は「month」や「quarter」、「car」がクエリの場合は「auto」や「vehicle」などの類義語が得られていることがわかる。
* このように単語の意味的な点と文法的な点において、似た単語同士が近いベクトルとして表されていることがわかる。


# word2vec
* ここまではカウントベースの手法によって、単語の分散表現を得てきたが、それに代わる強力な手法として推論ベースの手法がある。
* 推論ベースの手法ではニューラルネットワークを用いており、有名な手法としてword2vecがある。
* 以降ではword2vecの仕組みを理解し、実装することを行なう。
* ただまずは、シンプルなword2vecを実装することで処理効率よりもわかりやすさを重視した実装を行っていく。
* これまでは単語の分散表現(ベクトル化)の手法として、大きく「カウントベースの手法」と「推論ベースの手法」が研究されてきた。
* 両者のアプローチは大きくことなるが、どちらも分布仮説(単語自体に意味はなく、その単語のコンテキストによって、単語の意味が形成されるという仮説)に基づいている。
* まずここまで用いてきたカウントベースの手法では周囲の単語の頻度によって単語を表現してきた。
* 具体的には共起行列を作り、さらにSVDを適用することで密なベクトルを得てきた。
* ただ、このカウントベースの手法には大きな問題があり、その問題は規模が大きなコーパスを扱う際に顕在化する。
* 現実世界でコーパスを扱う場合、語彙数は非常に多くなり得る。
* 例えば、英語の場合、語彙数は100万を優に超えると言われている。
* この場合、カウントベースの手法では、100万×100万の巨大な行列を作ることになり、その巨大な行列に対して、SVDを行なうことは現実的ではないと言える。
* SVDはn×nの行列に対して、O(n^3)の計算コストがかかる。これはnの大きさの3乗に比例して計算時間が増えていくことを意味する。
* このような計算はスーパーコンピューターでも困難であるため、実際は近似的な手法や疎な行列の性質を利用して処理速度の向上させるのが一般的である。
* ただそのような高速化処理を行っても、多くの計算李s-すと計算時間が必要となる。
* カウントベースの手法はコーパス全体の統計データ(共起行列やPPMIなど)を利用して1回の処理(SVDなど)で単語の分散表現を得ている。
* 一方、推論ベースの手法では、ニューラルネットワークを用いる場合、ミニバッチで学習するのが一般的である。
* ミニバッチで学習するとは、1度に少量(ミニバッチ)の学習サンプルを使って、重みを繰り返し更新していくことを意味する。
* つまり、カウントベースの手法が学習データを1度にまとめて処理するのに対し、推論ベースの手法では、学習データの一部を使って逐次的に学習していると言える。
* これは語彙数が多いコーパスにおいて、SVDなどの計算量が膨大で難しい処理を行う場合でも、ニューラルネットワークではデータを小分けにして学習を行うことが可能なことを意味している。
* さらにニューラルネットワークの学習では、複数マシン・複数GPUの利用による並列計算も可能であり、全体の学習も高速化することができる。
* これらの点において、推論ベースの手法の方が優れていると言える。
* 推論ベースの手法では、以下のように周囲の単語が与えられた特に「？」にどんな単語が出現するかを推論する作業と言える。
    * you ? goodbye and I say hello.
* また、このような推論問題を解き、学習することが「推論ベースの手法」が扱う問題である。
* このような推論問題を繰り返し解くことで単語の出力パターンを学習する。
* つまり推論ベースの手法では何らかのモデルが登場し、そのモデルにニューラルネットワークを用いる。
* このモデルはコンテキスト情報を入力として受け取り、出現しうる各単語の出現する確率を出力する。
* モデルによって正しい推測ができるようにコーパスを用い、モデルの学習を行う。
* その学習の成果として単語の分散表現が得られる、というのが推論ベースの手法の全体図と言える。
* ニューラルネットワークを使って単語を処理する場合、"you"や"say"のような単語をそのまま処理することはできず、固定長のベクトルに変換する必要がある。
* 固定長べのベクトルへの変換手法の1つとして、one-hot表現がある。
* one-hot表現とは、ベクトル中の要素のうち、1つだけが1で残りは0であるベクトル表現のことを指す。
* "You say goodbye and I say hello."という1分をコーパスとして扱う場合を考える。
* このコーパスには語彙が7個存在し、この各単語のone-hot表現は以下のようになる。
    | 単語    | ID  | one-hot表現            |
    |:-------:|:---:|:---------------------:|
    | you     | 0   | (1, 0, 0, 0, 0, 0, 0) |
    | say     | 1   | (0, 1, 0, 0, 0, 0, 0) |
    | goodbye | 2   | (0, 0, 1, 0, 0, 0, 0) |
    | and     | 3   | (0, 0, 0, 1, 0, 0, 0) |
    | i       | 4   | (0, 0, 0, 0, 1, 0, 0) |
    | hello   | 5   | (0, 0, 0, 0, 0, 1, 0) |
    | .       | 6   | (0, 0, 0, 0, 0, 0, 1) |
* 上記の通り、単語のone-hot表現はコーパスの語彙数分の要素数を持つベクトルを用意し、単語IDに該当する要素を1、残りの要素を0として生成している。
* このように単語を固定長のベクトルに変換することで、ニューラルネットワークの入力層はニューロンの数を固定することができる。
* 上記のコーパスを学習データの対象とした場合、入力層は7つのニューロンで構成する。
* つまり、各ニューロンがそれぞれ7つの単語に対応する。
* これにより、単語をベクトルで表現することでその単語(ベクトル)はニューラルネットワークを構成する様々なレイヤで処理することができる。
* ここでone-hot表現で表された単語に対して全結合層で変換する場合を考える。
* 全てのノードには中間層とのつながりがありそこには重みが存在するため、中間層のニューロンは入力層のニューロンとの重み付き和となる。
* また、ここではニューロンのバイアスは省略する。
* バイアスを用いない全結合層は「行列の積」の計算に相当し、多くのディープラーニングのフレームワークでは全結合層の生成時にバイアスを用いない選択をすることが可能となっている。
* ここまでの全結合層による変換は以下のように実装される。　※中間層は3ノードとする
    ```python
    import numpy as np

    C = np.array([[1, 0, 0, 0, 0, 0, 0]])  # 入力
    W = np.random.randn(7, 3)              # 重み
    h = np.dot(c, W)                       # 中間ノード
    print(h)
    # [[0.70012195  0.25204755  0.79774592]]
    ```
* 上記では単語IDが0の単語をone-hot表現で表し、それを入力として、全結合層によって変換をしている。
* 全結合層の計算はバイアスを省略しているため、行列の積で行うことができ、Numpyのnp.dot()を用いている。
* また、cとWの積の部分はcがone-hot表現(特定の要素のみが1でそれ以外は0)のため、重みWの行ベクトル(=1行n列行列)を抜き出す作業に相当する。
    * 単語IDが0の場合は、one-hot表現されたベクトルcは0番目の要素のみが1である。
    * このベクトルと任意の値を要素として持っている重みWの積を計算した場合、cの0番目の要素との積を求める部分だけが0ではなく、Wの要素の値そのものになる。
    * ベクトルcの0番目の要素との積はWの0行目の要素と行われる。他の要素との積は行っても、0となるため、結果として、Wの0行目の要素を抜き出すことになる。
* 行ベクトルを抜き出すために行列計算を行うのは非効率だが、その改良に関しては後述する。
* また、この実装をMatMulレイヤを使用すると以下のようになる。
    ```python
    import sys
    sys.path.append('..')
    import numpy as np
    from common.layers import MatMul

    C = np.array([[1, 0, 0, 0, 0, 0, 0]])
    W = np.random.randn(7, 3)
    layer = MatMul(W)
    h = layer.forward(c)
    print(h)
    # [[0.70012195  0.25204755  0.79774592]]
    ```
* MatMulレイヤに重みWを設定し、forward()メソッドで順伝播の処理を行うことで、積を計算する。
* 以下ではシンプルなword2vecを実装する。つまり、「モデル」としてニューラルネットワークを組み込む。
* ここでは、word2vecで提案されているcontinuous bag-of-words(CBOW)というモデルを使用する。
    * word2vecという用語は本来、プログラムやツール類を指して用いられていたが、文脈によってはニューラルネットワークのモデルを指す場合もある。
    * 厳密にはCBOWモデルとskip-gramモデルという2つのモデルがword2vecで使用されるニューラルネットワークである。
* CBOWモデルはターゲット(中央の単語)をコンテキスト(周囲の単語)を推測することを目的としたニューラルネットワークである。
* CBOWモデルを正確な推測ができるように訓練することで単語の分散表現を獲得することができる。
* CBOWモデルへの入力はコンテキストであり、['you', 'goodbye']のような単語のリストで表されるが、これをone-hot表現に変換し、CBOWモデルが処理できるようにしてから用いる。
* CBOWモデルのネットワークは入力層を2つ(7ノード×2)とし、中間層(7ノード×1)を経て、出力層(7ノード×1)で構成される。
* 2つの入力層から中間層への変換は同じ全結合層によって行い、その時の重みはW_in(7×3行列)とする。
* また、中間層から出力層への変換は別の全結合層によって行い、その時の重みはW_out(3×7行列)とする。
* コンテキストとして2つの単語を考える場合、上記のように入力層が2つとなる。つまりN個のコンテキストを扱う場合、N個の入力層を用意する必要がある。
* 中間層の各ニューロンでは各入力層の全結合による変換後の値が平均されたものになる。
* つまり、入力層が2つの場合、それぞれの変換後の値がh1, h2とすると、中間層のニューロンは(h1+h2)/2となる。
* 出力層には7個のニューロンがあり、それらのニューロンは各単語に対応している。
* 出力層のニューロンは各単語の「スコア」であり、その値が高いほど、その単語の出現確率が高いと言える。
* ただし、ここでのスコアは確率として解釈される前の値となっており、実際はこのスコアにSoftmax関数を適用して、確率として値を取得する。
* 入力層から中間層への変換は全結合層によって行われるが、この全結合層の重みW_inは7×3の行列でこの重みが単語の分散表現となる。
* 重みW_inの各行には各単語の分散表現が格納されていると考え、学習を重ねることでコンテキストから出現する単語をうまく推測できるように各単語の分散表現が更新されていく。
* また、このようにして得られたベクトルは単語の意味もうまくエンコードされている。
* ここで中間層のニューロン数は入力層のニューロン数よりも少なくする必要がある。
* これは中間層には単語を予測するために必要な情報をコンパクトに収める必要があり、またその結果として密なベクトル表現が得られるためである。
* CBOWモデルの推論処理(スコアを算出する処理)の実装は以下の通り。
    ```python
    import sys
    sys.path.append(..)
    import numpy as np
    from common.layers import MatMul

    # Context data sample
    c0 = np.array([[1, 0, 0, 0, 0, 0, 0]])
    c1 = np.array([[0, 0, 1, 0, 0, 0, 0]])

    # Initialize wegiht
    W_in = np.random.randn(7, 3)
    W_out = np.random.randn(3, 7)

    # Create layer
    in_layer0 = MatMul(W_in)
    in_layer1 = MatMul(W_in)
    out_layer = MatMul(W_out)

    # Forward propagation
    h0 = in_layer0.forward(c0)
    h1 = in_layer0.forward(c1)
    h = 0.5 * (h0+h1)
    s = out_layer.forward(h)

    print(s)
    # [[ 0.30916255  0.45060817  -0.77308656  0.22054131  0.15037278
    #   -0.93659277 -0.59612048]]
    ```
* まず重みW_in、W_outをランダム値で初期化し、入力側のMatMulレイヤをコンテキストの数だけ(ここでは2つ)生成し、また出力層側のMatMulレイヤを1つ生成する。
* ここで、入力側のMatMulレイヤは重みW_inを共有して生成する。
* る偽にMatMulレイヤのforward()メソッドで中間データを生成し、出力層側のMatMulレイヤのforward()メソッドによって、各単語のスコアを算出する。
* 上記の通り、CBOWモデルは活性化関数を用いないシンプルな構成になっている。
* 次にCBOWモデルの学習を行う。
* まず、CBOWモデルは出力層において各単語のスコアを出力するが、そのスコアに対してSoftmax関数を適用することで、その言語の確率を得ることができる。
* この確率は前後の単語であるコンテキストが与えられた時にその中央にどの単語が出現するかを表すものとなる。
* 学習データとして、例えばコンテキストとして「you」、「goodbye」が与えられ、正解ラベルが「say」として与えられる。
* このとき、十分に学習された重みをもつネットワークがあれば、確率を表すニューロンにおいて、「say」に対応するニューロンの確率が高くなると言える。
* よって、CBOWモデルの学習は正しい予測ができるように重みを調整することを意味する。
* それにより、W_in、W_outに単語の出現パターンを捉えたベクトルが学習される。
* CBOWモデルの学習で得られる単語の分散表現は単語の意味的な点、文法的な点において、人間の直感と合致するケースが多いことがこれまでの研究から分かっている。
* CBOWモデルの学習ではコーパスにおける単語の出現パターンを学んでいるため、コーパスが異なれば、学習で得られる単語の分散表現も異なるものになる。
* 例えば、コーパスとして、スポーツのニュース記事だけを使う場合と音楽のニュース記事だけを使う場合では得られる単語の分散表現は大きく異なるものとなる。
* 上記の推論処理の実装に実装を追加して学習を行うためには、多クラス分類を行うためにSoftmax関数と交差エントロピー誤差を導入する必要がある。
* 上述の実装で得られるスコアをSoftmax関数で確率に変換し、その確率と正解データから交差エントロピー誤差を求め、それを損失として学習を行う。
* よって、CBOWモデルに対して、SoftmaxレイヤとCross Entropy Errorレイヤを追加することで学習が可能となるが、ここでは、以前に実装済みのSoftmax with Lossというレイヤを使用して実装する。
* word2vecで使用されるネットワークには上述の通り、入力側の全結合層の重みW_inと出力側の全結合層の重みW_outがある。
* W_inは各行が各単語の分散表現となっており、W_outは単語の意味がエンコードされたベクトルが格納されていると考えることができる。
* さらに出力側の重みW_outは列方向(縦方向の並び)に各単語の分散表現が格納されている。
* このように入力側・出力側の両方の重みに含まれる単語の分散表現はどちらを利用するべきか？を考える。
* 入力側だけ使用する、出力側だけ使用する、両方使用する、などが考えられる。
* また、両方使用する場合は、どのように2つを組み合わせるかでさらにいくつかの手法が考えられる。
* word2vecに関しては、入力側の重みだけを使用するのが最も用いられている方法である。
* いくつかの研究でW_inのみを使用することの有効性が示されている。
* 学習で用いる学習データはこれまでと同様、「You say goodbye and I say hello.」をコーパスとして使用する。
* word2vecで用いるニューラルネットワークの入力はコンテキストであり、正解データはコンテキストで囲まれた中央の単語(ターゲット)である。
* つまり、学習の目標はニューラルネットワークにコンテキストを入力したときにターゲットの単語が出現する確率を高くすることである。
* 上記を踏まえ、コーパスからコンテキストとターゲットを作ることを考える。
* 以下では両端の単語を除くコーパス中の全ての単語をターゲットとし、そのコンテキストを抜き出している。
    | contexts     | target  |
    |:-------------|:--------|
    | you, goodbye | say     |
    | say, and     | goodbye |
    | goodbye, I   | and     |
    | and, say     | I       |
    | I, hello     | say     |
    | say, .       | hello   |
* 上記のcontextsの各行がニューラルネットワークの入力となり、targetの各行が正解データとなる。
* まず学習を行う準備として、コーパスからコンテキストとターゲットを作成する関数を作成する。
* はじめにpreprocess()を用い、以下のようにコーパスのテキストを単語IDに変換する。
    ```python
    import sys
    sys.path.append('..')
    from common.util import preprocess

    test = 'You say goodbye and I say hello.'
    corpus, word_to_id, id_to_word = preprocess(text)
    print(corpus)
    # [0  1  2  3  4  5  6]

    print(id_to_word)
    # {0:'you', 1:'say', 2:'goodbye', 3:'and', 4:'i', 5:'hello', 6:'.'}
    ```
* 次にこの単語IDの配列から上記で行ったのと同様、contextsとtargetを抜き出すことを単語IDベースで行い、それぞれを単語IDのリストとして格納する。
    | contexts  | target  |
    |:----------|:--------|
    | [[0  2]   | [1      |
    |  [1  3]   |  2      |
    |  [2  4]   |  3      |
    |  [3  1]   |  4      |
    |  [4  5]   |  1      |
    |  [1  6]]  |  5]     |
* これにより、コンテキストは6×2の2次元配列(次元はコンテキストの大きさ)、ターゲットは6個の要素を持つ1次元の配列となる。
* 以上を踏まえ、コーパスからコンテキストとターゲットを作成する関数create_contexts_target()を実装すると以下のようになる。
    ```python
    def create_contexts_target(corpus, window_size=1):
        target = corpus[window_size:-window_size]　#corpus[1:-1]
        contexts = []

        for idx in range(window_size, len(corpus)-window_size):
            cs = []
            for t in range(-window_size, window_size+1):　# -1, 0, 1
                if t == 0:
                    continue
                cs.append(corpus[idx+t])
            contexts.append(cs)
        
        return np.array(contexts), np.array(target)
    ```
* まず、targetをcorpusの両端の単語以外をリストとして持つため、corpus[window_size:-window_size]として格納する。
* 次にfor文を入れ子で回す。外のfor文でcorpusを両端の単語を除く先頭から回し、内のfor文でtargetとその前後のコンテキストを回し、target以外の要素を取得する処理を行う。
* 関数としては、返り値をcontextsとtargetをそれぞれNumpyの多次元配列として返している。
* この関数を使う実装は以下のようになる。
    ```python
    contexts, target = create_contexts_target(corpus, window_size=1)

    print(contexts)
    # [[0  2]
    #  [1  3]
    #  [2  4]
    #  [3  1]
    #  [4  5]
    #  [1  6]]

    print(target)
    # [1  2  3  4  1  5]
    ```
* 上記により、単語IDのコーパスからコンテキストとターゲットを生成することができた。
* 次にこれらをCBOWモデルに渡すために単語IDで構成されているコンテキストとターゲットをone-hot表現に変換する。
    | contexts            | target            |
    |:--------------------|:------------------|
    | [[[1 0 0 0 0 0 0]   | [[0 1 0 0 0 0 0]  |  
    |   [0 0 1 0 0 0 0]]  |                   |
    |  [[0 1 0 0 0 0 0]   |  [0 0 1 0 0 0 0]  |
    |   [0 0 0 1 0 0 0]]  |                   |
    |  [[0 0 1 0 0 0 0]   |  [0 0 0 1 0 0 0]  |
    |   [0 0 0 0 1 0 0]   |                   |
    |  [[0 0 0 1 0 0 0]   |  [0 0 0 0 1 0 0]  |
    |   [0 1 0 0 0 0 0]]  |                   |
    |  [[0 0 0 0 1 0 0]   |  [0 1 0 0 0 0 0]  |
    |   [0 0 0 0 0 1 0]]  |                   |
    |  [[0 1 0 0 0 0 0]   |  [0 0 0 0 0 1 0]] |
    |   [0 0 0 0 0 0 1]]] |                   |
* これにより、コンテキストは6×2×7の3次元配列、ターゲットは6×7の2次元配列となる。
* このone-hot表現への変換のconvert_one_hot()関数を用いて以下のような実装となる。
    ```python
    import sys
    sys.path.append('..')
    from common.util import preprocess, create_contexts_target, convert_one_hot

    text = 'You say goodbye and I say hello.'
    corpus, word_to_id, id_to_word = preprocess(text)

    contexts, target = create_contexts_target(corpus, window_size=1)

    vocab_size = len(word_to_id)
    target = convert_one_hot(target, vocab_size)
    contexts = convert_one_hot(contexts, vocab_size)
    ```
* ここまででCBOWモデルの学習のための準備が完了し、以下では学習に関する実装を行う。
* 実装するクラスはSimpleCBOWとしての実装は以下の通りとなる。
    ```python
    import sys
    sys.path.append('..')
    import numpy as np
    from common.layers import MatMul, SoftmaxWithLoss

    class SimpleCBOW:
        def __init__(self, vocab_size, hidden_size):
            V, H = vocab_size, hidden_size

            # Initialize weight
            W_in = 0.01 * np.random.randn(V, H).astype('f')
            W_out = 0.01 * np.random.randn(H, V).astype('f')

            # Create layers
            self.in_layer0 = MatMul(W_in)
            self.in_layer1 = MatMul(W_in)
            self.out_layer = MatMul(W_out)
            self.loss_layer = SoftmaxWithLoss()

            # Create lists of all weight and gradient
            layers = [self.in_layer0, self.in_layer1, self.out_layer]
            self.params, self.grads = [], []
            for layer in layers:
                self.params += layer.params
                self.grads += layer.grads
            
            # Set the distributed expression of words to member variables.
            self.word_vecs = W_in
    ```
* まずコンストラクタは上記のような実装となる。
* 引数として語彙数vocab_sizeと中間層のニューロンの数hidden_sizeを受け取る。
* 重みの初期値として、W_in、W_outを生成し、NumPy配列のデータ型astype('f')とし、要素を32ビットの浮動小数点数とする。
* 次に入力側のMatMulレイヤを2つ、出力側のMatMulレイヤを1つ、Softmax with Lossレイヤを1つそれぞれ作成する。
* 入力側のMatMulレイヤ数はコンテキストで使用する単語数と同じ分だけ生成する(ここでは2つ)
* 次にニューラルネットワークで使われるパラメータと勾配をメンバー変数params, gradsにリストとして格納する。
* また、W_in(単語の分散表現)もメンバーword_vecsに格納する。
* 上記の実装だと同じ重みを複数のレイヤで共有していることになるため、メンバー変数paramsのリストには、同じ重みが複数存在することになる。
* これにより、AdamやMomentumなどのoptimize処理が正しく動作しなくなる可能性がある。
* よって、Trainerクラス内部でパラメータの更新時にパラメータの重複を取り除く処理を行っている。
* 次にニューラルネットワークの順伝播処理を行うメソッドforward()を実装すると以下の通りとなる。
    ```python
        def forward(self, contexts, target):
            h0 = self.in_layer0.forward(contexts[:, 0])
            h1 = self.in_layer1.forward(contexts[:, 1])
            h = (h0 + h1) * 0.5
            score = self.out_layer.forward(h)
            loss = self.loss_layer.forward(score, target)
            return loss
    ```
* forward()メソッドは引数として、contextsとtargetの2つを取り、損失(loss)を返す。
* contextsはここでは3次元のNumPy配列となり、(6,2,7)=(コンテキストペアの数,コンテキストサイズ,語彙数)となる。
* また、targetはここでは2次元NumPy配列となり、(6,7)=(コンテキストペアの数,語彙数)となる。
* 次にニューラルネットワークの逆伝播処理を行うメソッドbackward()を実装すると以下の通りとなる。
    ```python
        def backward(self, dout=1):
            ds = self.loss_layer.backward(dout)
            da = self.out_layer.backward(ds)
            da *= 0.5
            self.in_layer1.backward(da)
            self.in_layer0.backward(da)
            return None
    ```
* 各パラメータの勾配をメンバー変数gradsにまとめ、順伝播・逆伝播の実装を行なったため、forward()メソッドを呼び、次にbackward()メソッドを呼ぶことでgradsに保持されている勾配が更新することができる。
* 以下では、SimpleCBOWクラスの学習の実装を行う。
* CBOWモデルの学習も通常のニューラルネットワークの学習と全く同じで、まず学習データを準備してニューラルネットワークに与え、勾配を算出し、重みを逐一アップデートしていく処理を行う。
    ```python
    import sys
    sys.path.append('..')
    from common.trainer import Trainer
    from common.optimizer import Adam
    from simple_cbow import SimpleCBOW
    from common.util import preprocess, create_contexts_target, convert_one_hot

    window_size = 1
    hidden_size = 5
    batch_size = 3
    max_epoch = 1000

    text = 'You say goodbye and I say hello.'
    corpus, word_to_id, id_to_word = preprocess(text)

    vocab_size = len(word_to_id)
    contexts, target = create_contexts_target(corpus, window_size=1)
    target = convert_one_hot(target, vocab_size)
    contexts = convert_one_hot(contexts, vocab_size)

    model = SimpleCBOW(vocal_size, hidden_size)
    optimizer = Adam()
    trainer = Trainer(model, optimizer)

    trainer.fit(contexts, target, max_epoch, batch_size)
    trainer.plot()
    ```
* パラメータを更新するoptimizerには様々な手法があるが、ここではAdamを使用する。
* Trainerクラスではニューラルネットワークの学習を行なう。
* 具体的には学習データからミニバッチを生成し、それをニューラルネットワークに与えて勾配を算出、その勾配をoptimizerに渡してパラメータを更新する処理を行う。
* このような一連の処理をTrainerクラスに集約することで学習のためのコードをシンプルにすることができる。
* 最後のplot()メソッドで実行結果をグラフで確認する。
* 縦軸を損失、横軸をiterationとしてグラフを表示すると、学習を重ねるごとに損失が減少していくことがわかる。
* ここで、学習が終わった後の重みパラメータを確認する。
* 入力側のMatMulレイヤの重みを以下のコードで取り出す。
    ```python
    word_vecs = model.word_vecs
    for word_id, word in id_to_word.items():
        print(word, word_vecs[word_id])
    ```
* 入力側のMatMulレイヤの重みはSimpleCBOWのメンバー変数word_vecsに格納されているので、これを取得し、for文で単語とその単語のベクトル表現を取得する。
* 実行結果は以下のようになる。
    ```
    you [-0.9031807  -1.0374491  -1.4682057  -1.3216232  0.93127245]
    say [ 1.2172916   1.2620505  -0.07845993  0.07709391 -1.2389531]
    goodbye [-1.0834033  -0.8826921  -0.33428606  -0.5720131  1.0488235]
    and [ 1.0244362  1.0160093  -1.6284224  -1.6400533  -1.0564581]
    i [ -1.0642933  -0.9162385  -0.31357735 -0.5730831  1.041875]
    hello [-0.9018145  -1.035476  -1.4629668  -1.3058501  0.9280102]
    . [ 1.0985303  1.1642815  1.4365371  1.3974973  -1.0714306]
    ```
* これにより、コーパスに含まれる単語の密なベクトル表現を得ることができたと言える。
* ただ、これらは'You say goodbye and I say hello.'という小さなコーパスから得られた結果に過ぎない。
* より大きく実用的なコーパスを使用することで、よりよい単語の分散表現を得ることができる。
* その場合、これまでのやり方だと処理速度の点で問題が発生する。これまでの手法には処理効率の点でいくつかの問題があるためである。
* 以降ではこれまでのシンプルなCBOWモデルに対して改良を加え、実用的なCBOWモデルを実装していく。
* まず、CBOWモデルを「確率」という視点から見てみる。
* 「確率」はP(・)と表され、Aという事象が起こる確率はP(A)と表記される。
* また、「AとBが同時に起こる確率」つまり同時確率はP(A,B)と表記し、「Bが起こった後にAが起こる確率」つまり事後確率はP(A|B)と表記される。事後確率は「Bという情報が与えられたときにAが起こる確率」と解釈することもできる。
* これらを踏まえ、CBOWモデルを確率の表記で記述することを考える。
* CBOWモデルはコンテキストを与えると、ターゲットとなる単語の確率を出力するという処理を行うので、ここでは、w1, w2,...wTという単語列のコーパスとウィンドウサイズが1のコンテキストを考える。
* ここで、コーパス中のt番目の単語をwtをすると、コーパスは以下のように表される。
    ```
    w1  w2  w3 ...  wt-1  wt  wt+1  ...  wT-1  wT
    ```
* このとき、「コンテキストとしてwt-1とwt+1が与えられたとき(=B)」に「ターゲットがwtになる(=A)」確率は事後確率を用いて以下のようになる。
    * P(wt | wt-1,wt+1)　　(3.1)
* これはつまり、CBOWは(3.1)をモデル化していることを意味する。
* また、(3.1)を用いることで、CBOWモデルの損失関数も簡潔に表すことができる。
* まず、交差エントロピー誤差は以下のように表すことができる。
    * L = - Σ tk・logyk　　(1.7)
        * yk：k番目に対応する事象が起こる確率
        * tk：教師ラベル(one-hotベクトル)
* このエントロピー誤差に「ターゲットがwtになる確率」をあてはめると、tkのt番目の要素のみが1、それ以外の要素は0となり、yt = P(wt | wt-1,wt+1)とすることができる。
* これを考慮すると、(3.1)、(1.7)から以下のようになる。
    * L = -logP(wt | wt-1,wt+1)　　(3.2)
* 上記より、CBOWモデルの損失関数は(3.1)の確率に対してlogを取り、その値にマイナスをつけたものであると言える。これは負の対数尤度(negative log likelihood)と呼ぶ。
* (3.2)は1つのサンプルデータに関する損失関数となるが、これをコーパス全体に拡張すると損失関数は以下のようになる。
    * L = -(1/T)・ΣlogP(wt | wt-1,wt+1)　　※t = 1, 2, ..., T　　(3.3)
* これを踏まえると、CBOWモデルの学習とはこの(3.3)で表される損失関数を小さくすることであり、重みパラメータが単語の分散表現となる。
* word2vecでは、CBOWモデルの他にskip-gramと呼ばれるモデルがある。
* skip-gramはCBOWで扱うコンテキストとターゲットを逆転させたモデルである。
* CBOWモデルとskip-gramモデルで解くべき問題を比較すると以下のようになる。
    * CBOWモデル：　you  ?  goodbye and I say hello.
    * skip-gramモデル：　?  say  ?  and I say hello.
* CBOWモデルはコンテキストが複数あり、それらから中央のターゲットの単語を推測するのに対し、skip-gramモデルは中央のターゲットの単語から周囲の複数の単語(コンテキスト)を推測する。
* よって、skip-gramモデルのレイヤー構成はCBOWモデルのレイヤー構成の逆の構成になる。
* つまり、skip-gramモデルでは入力層が1つで出力層がコンテキストの数だけ存在するような構成となる。
* 出力層では個別に損失を求め、それらを合計したものを最終的な損失とする。
* 次にskip-gramモデルを確率の表記を用いて表すことを考える。
* ターゲットとなる単語をwt、ターゲットの単語のコンテキストとなる単語をwt-1、wt+1とすると、skip-gramモデルは以下のように表される。
    * P(wt-1,wt+1 | wt)　　(3.4)
* これは「ターゲットwtが与えられたときにwt-1、wt+1が同時に起こる確率」を意味する。
* ここでは、wt-1、wt+1の間に関連性がないと仮定して、以下のように分解した表現として考える。
    * P(wt-1,wt+1 | wt) = P(wt-1 | wt) P(wt+1 | wt)　　(3.5)
* さらにCBOWモデルと同様、(1.7)の交差エントロピー誤差の式を適用し、skip-gramモデルの損失関数を導くと以下のようになる。
    * L = -logP(wt-1,wt+1 | wt) = -logP(wt-1 | wt) P(wt+1 | wt) = -(logP(wt-1 | wt) + logP(wt+1 | wt))　　(3.6)
    * ※ここでは、logxy = logx + logyの性質を用いて変換している。
* 上式の通り、skip-gramモデルの損失関数はコンテキスト分の損失をそれぞれ求め、合計したものであると言える。
* CBOWモデルと同様、損失関数をコーパス全体に拡張すると、以下のようになる。
    * L = -(1/T)Σ(logP(wt-1 | wt) + logP(wt+1 | wt))　　※t = 1, 2, ..., T　　(3.7)
* (3.3)、(3.7)を比較することで、CBOWモデルとskip-gramモデルの違いがわかる。
* skip-gramモデルはコンテキストの数だけ推測するため、損失関数は各コンテキストで求めた損失の総和を求めているのに対し、CBOWモデルは1つのターゲットの損失を求めている。
* これまでのCBOWモデルとskip-gramモデルでは、skip-gramモデルを使用する方がよい。
* skip-gramモデルの方が、多くの場合、単語の分散表現の精度がよいこと、またコーパスが大規模になると、低頻出単語や類推問題の性能において、優れた結果が得られる傾向があることが知られているからである。
* 一方、学習速度の点ではCBOWモデルの方が、skip-gramモデルよりも高速である。
* これはskip-gramモデルはコンテキストの数だけ損失を求めるため、計算コストが大きくなるためである。
* skip-gramモデルは1つの単語からその周囲の単語を予測することになるが、直感的にこの問題は難しいと言える。
* CBOWモデルではコンテキストから1つの単語を予測するので、直感的には比較的簡単な問題と言える。
* つまり、skip-gramモデルの方がより難しい問題に取り組み、学習しているため、より優れた単語の分散表現を得られると言える。
* 次のカウントベースの手法と推論ベースの手法(特にword2vec)の手法を比較する。
* カウントベースの手法はコーパス全体の統計データから1回の学習で単語の分散表現を得ていたが、推論ベースの手法では、コーパスの一部を何度も見ながらミニバッチとして学習し、単語の分散表現を得ている。
* まず、語彙に新しい単語を追加するケースで単語の分散表現の更新作業が発生する場合を考える。
* この場合、カウントベースの手法では、ゼロから計算をし直す必要がある。
* 単語の分散表現を少し修正したい場合であっても、再度、共起行列を作り直し、SVDを行なう一連の作業が必要となる。
* 一方、推論ベースの手法であるword2vecでは、パラメータの再学習を行うことができる。
* つまり、これまで学習した重みを初期値として再学習することでこれまでの学習経験を失うことなく、単語の分散表現の更新を効率的に行うことができる。
* これらを踏まえると、語彙が追加された場合は推論ベースの手法の方がよいと言える。
* 次にカウントベースの手法と推論ベースの手法それぞれで得られる単語の分散表現の性質や精度を比較する。
* 分散表現の性質に関しては、カウントベースの手法では主に単語の類似性がエンコードされることがわかっている。
* word2vecでは単語の類似性に加え、さらに複雑な単語間のパターンも捉えることができることがわかっている。
    * word2vecで「king - man + woman = queen」のような類推問題を解けることが知られている。
* ただ単語の類似性に関する定量評価ではカウントベースの手法と推論ベースの手法に関しては、優劣をつけられないことが判明している。
* これはハイパーパラメータの依存度が大きいためである。
* また、推論ベースの手法とカウントベースの手法には関連性があることがわかっている。
* 具体的にはskip-gramモデル、negative samplingを用いたモデルは、カウントベースの手法で生成するコーパス全体の共起行列に対して、特殊な行列分解をしているのと同じであることが示されている。
* また、word2vec以降、推論ベースとカウントベースの手法を融合させたGloVeという手法も提案されている。
* この手法では、コーパス全体の統計データの情報を損失関数に取り入れてミニバッチ学習を行っている。


# word2vecの高速化
* これまではword2vecの仕組みを踏まえ、2層のシンプルなCBOWモデルを実装してきた。
* この実装には問題があり、特にコーパスで扱う語彙数が増えるに従い、計算量が増加する点が挙げられる。
* ある程度の語彙数に達すると、これまでのCBOWモデルでは計算量があまりにも大きくなりすぎてしまう。
* 以降では、word2vecの高速化に主眼を置き、word2vecの改善に取り組む。
* 具体的にはEmbeddingレイヤという新しいレイヤを導入すること、Negative Samplingという新しい損失関数を導入することの2点を行なう。
* これまでのCBOWモデルは、2つの単語をコンテキストとして処理し、それをもとに1つの単語をターゲットとして推測するモデルである。
* このとき、入力側の重みW_inとの積によって中間層が計算され、出力側の重みW_outとの行列の積によって各単語のスコアが算出される。
* 最後にそのスコアにSoftmax関数を適用することで各単語の出現確率が得られ、それを正解ラベルと比較する(交差エントロピー誤差を適用する)ことで損失を求めている。
* このようなモデルは小さなコーパスを扱う分には問題はないが、巨大なコーパスを扱う場合、いくつかの問題が発生する。
* 例えば、語彙数が100万、中間層のニューロン数が100の場合におけるCBOWモデルを考える。
* この場合、入力層側の重みWinは1000000×100となり、その行列が2つ必要となる。
* さらに出力層側の重みWoutも100×1000000の大きさとなる。
* このような巨大なニューロンにおける計算には特に以下の2箇所がボトルネックとなる。
    * 入力層のone-hot表現と重み行列W_inの積の計算
        * 単語をone-hot表現で扱っているため、語彙数が増えるにつれ、one-hot表現のベクトルサイズも増加する。
        * 語彙数が100万の場合、そのone-hot表現だけでも100万の要素を持つベクトルのメモリサイズが必要になる。
        * さらにそのone-hotベクトルと重み行列W_inの積を計算する際にさらに多くの計算リソースが必要となる。
        * この問題は後述のEmbeddingレイヤの導入により解決する。
    * 中間層と重み行列W_outの積の計算及びSoftmaxレイヤの計算
        * 中間層と重み行列W_outの積の計算に多くの計算リソースが必要となる。
        * 加えてSoftmaxレイヤに関する計算も扱う語彙数が増えるにつれて計算量が増加する。
        * この問題は後述のNegative Samplingという新しい損失関数を導入することにより解決する。


# word2vecの高速化 - Embeddingレイヤ
* これまでのword2vecの実装では単語をone-hot表現に変換し、それをMatMulレイヤに入力してMatMulレイヤ内でone-hotベクトルと重み行列の積を計算していた。
* ここで単語の語彙数が100万、中間層のニューロン数を100とすると、MatMulレイヤ内での行列の積の計算は以下のようになる。
    * one-hot表現：(1×1000000)
    * 重みW_in：(1000000×100)
    * 中間層h：(1×100)
* つまり、100万語の語彙からなるコーパスがあるとすると、単語のone-hot表現の次元数も100万となり、その巨大なベクトルと重み行列の積を計算することになる。
* ここで単語のone-hot表現とW_inの積の詳細を考えると、one-hot表現ベクトルは1つの要素のみ1であり、残りの要素の値は0である。
* よって、W_inとの積は結果として、W_inの特定の行(=one-hot表現で1となっている行)を抜き出す処理となる。
    * one-hot表現で5番目の要素が1となっている場合、抜き出すのはW_inの5行目の行となる。
* これを踏まえ、重みパラメータW_inから単豪IDに該当する行(ベクトル)を抜き出しレイヤ(Embeddingレイヤ)を作る。
    * Embeddingはword embedding(=単語の埋め込み)に由来する。
    * つまり、Embeddingレイヤに単語の埋め込み(単語の分散表現)が格納される。
    * 自然言語処理の分野において単語の分散表現(distributed representation)は、単語の埋め込み(word embedding)とも呼ばれる。
* 行列から特定の行を抜き出す処理は比較的容易に行うことができる。
* 例えば、重みWがNUmpyの2次元配列とすると、この重みから特定の行を抜き出すのは、以下のように"W[2]"というように書くだけである。
    ```python
    >>> import numpy as np
    >>> W = np.arange(21).reshape(7, 3)
    >>> W
    array([[  0,  1,  2],
           [  3,  4,  5],
           [  6,  7,  8],
           [  9, 10, 11],
           [ 12, 13, 14],
           [ 15, 16, 17],
           [ 18, 19, 20]])
    >>> W[2]
    array([  6,  7,  8])
    ```
* また、重みWから複数行の要素をまとめて抽出することも以下のように抽出する行番号のリストを渡すことで容易にできる。
    ```python
    >>> idx = np.array([1, 0, 3, 0])
    >>> W[idx]
    array([[ 3,  4,  5],
           [ 0,  1,  2],
           [ 9, 10, 11],
           [ 0,  1,  2]])
    ```
* これらを踏まえると、Embeddingレイヤのforward()メソッドの実装は以下のようになる。
    ```python
    class Embedding:
        def __init__(self, W):
            self.params = [W]
            self.grads = [np.zeros_like(W)]
            self.idx = None
        
        def forward(self, idx):
            W, = self.params
            self.idx = idx
            out = W[idx]
            return out
    ```
* 次にEmbeddingレイヤのbackward()メソッドを実装する。
* 順伝播では重みWの特定の行を抜き出すだけの処理を行っているが、これは単に特定の行に対応するニューロンだけを手を加えずに次の層に流していることになる。
* そのため、逆伝播では前層(出力側の層)から伝わってきた勾配を次層(入力側の層)へそのまま流すだけの処理になると言える。
* ただし、前層から伝わる勾配を重みの勾配dwの特定の行(idx)のみに設定するようにする必要がある。
* 以上を踏まえると、Embeddingレイヤのbackward()メソッドの実装は以下のようになる。
    ```python
        def backward(self, dout):
            dW, = self.grads
            dW[...] = 0
            dW[self.idx] = dout
            return None
    ```
* 上記の実装ではまず、重みの勾配を取り出し、dW[...] = 0でdwの要素を0に設定し直している。
* その後、前層から受け取ったdoutをidxで指定された行に代入する。これにより、doutはidx行以外の要素は全て0となる行列になる。
* ここでの実装は重みWと同じ大きさの行列dWを作成し、その該当する行に伝わってくる勾配を代入しているが、少し非効率な実装となっている。
* 重みWを更新するためにはdWのようにWと同じ大きさの大きな行列を作る必要はなく、更新したい行番号(idx)とその勾配(dout)を保持しておけば、それらの情報から重みWを更新できる。
* また、上記のbackward()の実装には1つ問題がある。
* それはidxが[0, 2, 0, 4]のように要素が重複している場合に発生する。
* この場合、dW[self.idx] = doutの処理で、idxが上記の場合はdWの0行目に2つの値が代入されることになり、どちらかの値が上書きされることになる。
* このため、この処理はdWへの代入ではなく、加算を行なう処理に変更する必要がある。これを踏まえた実装は以下の通り。
    ```python
        def backward(self, dout):
            dW, = self.grads
            dW[...] = 0

            for i, word_id in enumerate(self.idx):
                dW[word_id] += dout[i]
            # np.add.at(dW, self.idx, dout)

            return None
    ```
* 上記の通り、for文を使って該当するインデックスに勾配を加算する処理を行う。これにより、idxに重複するインデックスがあっても、正しく処理できる。
* また、この処理はNumpyのnp.add.at()で行うこともできる。np.add.at(A, idx, B)でBをAに加算するが、idxで加算するAのインデックスをidxで指定することができる。
* 一般にPythonでfor文を使用するより、Numpyのメソッドを使用する方が高速に処理できる。これはNumpyのメソッドが低レイヤにおいて様々な高速化や処理効率向上が行われているためである。
* 上記で実装したEmbeddingレイヤをWord2vecの実装の入力側のMatMulレイヤの代わりに用いることで、メモリ使用量を減らし無駄な計算を省くことができる。


# word2vecの高速化 - Negative Sampling
* 続いて、中間層以降の処理(行列の積とSoftmaxレイヤの計算部分)の改善を行うことを考える。
* ここでは、Negative Sampling(負例サンプリング)と呼ばれる手法を用いる。
* Softmaxの代わりにNegative Samplingを用いることで、語彙数が多くなっても計算量を少なく一定に抑えることができる。
* まず、中間層以降の計算の問題点を認識する上で、これまでと同様、語彙数が100万、中間層のニューロン数が100のword2vec(CBOWモデル)を考える。
* この場合、入力層と出力層にそれぞれ100万個のニューロンが存在することになる。
* このときの中間層以降の処理で多くの計算時間が必要となる処理は以下の2つになる。
    * 中間層のニューロンと重み行列(W_out)の積
        * 上記の場合、中間層のベクトルサイズが100、W_outは100×100万の行列となるが、この巨大な行列同士の積の計算には多くの時間がかかり、また多くのメモリも使用することになる。
        * また、逆伝播時も同様の計算を行なうため、これらの行列の積の計算を軽くする必要がある。
    * Softmaxレイヤの計算
        * 語彙数が増えるに従い、Softmaxの計算量が増加する。k番目の単語(要素)を対象としたSoftmaxの式は以下の通り。
            * yk = (exp(sk)) / (Σexp(si))　※i=1,2, ... 1000000　　(4.1)
            * si：各単語のスコア
        * 上式の通り、語彙数が増えるに伴い、expの計算が増えることがわかる。
* Negative samplingの手法のキーは「二値分類」にあり、「多値分類」を「二値分類」で近似することにある。
* これまで行ってきた100万個の単語の中から正しい単語を1つ選ぶ問題は多値分類問題であり、それらを二値分類として近似し、二値分類として扱うことができないかを考える。
* これまではコンテキストを与え、正解となる単語を高い確率で推測できること目指し、ニューラルネットワークの学習を行ってきた。
* これにより、うまく学習ができれば、そのニューラルネットワークによって、正しい推測ができるようになると言える。
* 具体的にはコンテキストが「you」と「goodbye」のとき、ターゲットとなる単語が何になるか？をニューラルネットワークによって推測することになる。
* ここで多値分類を二値分類で近似するということは、Yes/Noで答えを出せる問題に変換することであり、コンテキストが「you」と「goodbye」のとき、ターゲットとなる単語は「say」になるか？を推測する問題となる。
* この場合、出力層はニューロンがsayであるスコアを出力するもの1つだけで事足りることになる。
* 出力層のニューロンが1つの場合のCBOWモデルは以下のようになる。
    * W_in(1000000×100)が2つ　※コンテキストのウィンドウサイズが1の場合
    * W_out(100×1)
* 出力層のニューロンが1つだけとなるため、中間層ベクトルと出力側の重み行列W_outとの積はW_outから「say」に対応する列(単語ベクトル)だけを抽出し、その抽出したベクトルと中間層のニューロンとの内積を計算するだけとなる。
* この内積の結果が最終的なスコアとなる。
* これにより得られたスコアはシグモイド関数を適用し、確率を取得する。
* 多値分類問題で確率を得るために使用していたSoftmax関数に対し、二値分類問題での確率を得るためにはシグモイド関数を使用する。
* また、二値分類問題における損失関数として、多値分類問題と同様、交差エントロピー誤差を用いる。
* シグモイド関数は以下の式で表される関数である。
    * y = 1 / (1+exp(-x))　　(4.2)
* シグモイド関数はS字カーブをしており、入力された値xは0～1の間の実数へと変換される。
* よって、シグモイド関数の出力yを確率として解釈することができる。
* 確率yから損失を求める場合、以下のような交差エントロピー誤差を用いる。
    * L = -(t logy +(1-t)log(1-y))　　(4.3)
        * y：シグモイド関数が出力する確率(=ニューラルネットワークが出力する確率)
        * t：正解ラベル(0 or 1)
* t=1のとき、正解は「Yes」であり、上式から損失Lは-logyとなる。
* t=0のとき、正解は「No」であり、上式から損失Lは-log(1-y)となる。
* (1.7)と(4.3)はそれぞれ多値分類問題と二値分類問題の交差エントロピー誤差を表している。
* (1.7)において、出力層に2つのニューロンを利用するとした場合、(1.7)と同じとなる。
* ここで、SigmoidレイヤとCross Entropy Errorレイヤの計算グラフを考えると以下の通りとなる。
    * 順伝播
        * x：Sigmoidレイヤへの入力
        * y：Sigmoidレイヤの出力(=Cross Entropy Errorレイヤへの入力)
        * Loss：Cross Entropy Errorレイヤの出力(=損失の値)
    * 逆伝播
        * dL/dy：Cross Entropy Errorレイヤの出力(Sigmoidレイヤへの入力)
        * dL/dx = y-t：Sigmoidレイヤの出力
* 上記の通り、逆伝播の出力はy-tとなる。
    * 逆伝播の値が同様にy-tになるのは、以下のような場合である。
        * シグモイド関数と交差エントロピー誤差
        * ソフトマックス関数と交差エントロピー誤差
        * 恒等関数と2乗和誤差
* これは例えば、正解ラベルが1のとき、yができるだけ1に近づくことでその誤差が小さくなることを意味する。
* 逆にyが1から遠ざかることでその誤差が大きくなることを意味する。
* これらの誤差が前層に流れることで、誤差が大きい場合は大きく学習し、誤差が小さい場合は小さく学習するようになる。
* ここまでの流れを整理すると以下のようになる。
    * これまでは出力層に語彙数分のニューロンを用意し、それをSoftmaxレイヤを通すことで多値分類問題を扱うことを行ってきた。
    * 入力層ではMatMulレイヤからEmbeddingレイヤに変更することで、各単語に対応する単語IDの分散表現をW_inから抜き出すことで処理の高速化・効率化を図る。
    * 次にニューラルネットワークを多値分類を行うネットワークから二値分類を行うネットワークに変換する。
    * ここでは中間層のニューロン(ベクトル)をhとし、出力側の重みW_outのうち、該当の単語に対する単語ベクトルとの内積を計算する。
    * 取得された内積をスコアとし、Sigmoid with lossレイヤに通すことで損失の値を得る。
* ここで計算グラフをシンプルにするためにEmbedding Dotレイヤを導入する。
* Embedding Dotレイヤは前述のEmbeddingレイヤとdot演算(内積)の2つの処理を合わせたレイヤである。
* これを中間層ベクトルとW_outの該当単語ベクトルとの内積を計算する処理に置き換える。
* Embedding Dotレイヤの実装は以下の通りとなる。
    ```python
    class EmbeddingDot:
        def __init__(self, W):
            self.embed = Embedding(W)
            self.params = self.embed.params
            self.grads = self.embed.grads
            self.cache = None
        
        def forward(self, h, idx):
            target_W = self.embed.forward(idx)
            out = np.sum(target_W * h, axis=1)

            self.cache = (h, target_W)
            return out
        
        def backward(self, dout):
            h, target_W = self.cache
            dout = dout.reshape(dout.shape[0], 1)

            dtarget_W = dout * h
            self.embed.backward(dtarget_W)
            dh = dout * target_W
            return dh
    ```
* EmbeddingDotクラスにはembed, params, grads, cacheの4つのメンバー変数があり、それぞれEmbeddingレイヤインスタンス、重みパラメータ、勾配、順伝播時の計算結果を格納する。
* forward()メソッドは引数として中間層のニューロンh、および単語IDのNumpy配列idxを受け取る。
* idxは単語IDの配列だが、これはデータをまとめて処理するミニバッチ処理を想定している。
* 内部の処理ではEmbeddingクラスのforward()メソッドを呼び、内積計算をnum.sum(target_W*h, axis=1)によって行っている。
* ここまでの処理を具体的な数値を使って表現すると以下の通りとなる。
    | W          | idx      | target_W   | h           | target_W*h | out       |
    |:-----------|:---------|:-----------|:------------|:-----------|:----------|
    |[[ 0  1  2] |[ 0  3  1]|[[ 0  1  2] |[[ 0  1  2]  |[[ 0  1  4] |[ 5 122 86]|
    | [ 3  4  5] |          | [ 9 10 11] | [ 3  4  5]  | [27 40 55] |           |
    | [ 6  7  8] |          | [ 3  4  5]]| [ 6  7  8]] | [18 28 40]]|           |
    | [ 9 10 11] |          |            |             |            |           |
    | [12 13 14] |          |            |             |            |           |
    | [15 16 17] |          |            |             |            |           |
    | [18 19 20]]|          |            |             |            |           |
* 上記では、まず適当な値の要素を持つ、Wとh、idxを用意する。
* ここでidxは[0, 3, 1]としており、この3つのデータをミニバッチとして扱うことを意味する。
* idxの要素により、　target_WはWの0番目、3番目、1番目の行の要素を抜き出したものになる。
* target_W*hで対応する各要素の積を計算し、その結果をaxis=1により、行ごとに合計してoutを算出する。
* 逆伝播処理を行うbackward()メソッドでは、順伝播とは逆順に勾配を前層に伝達する処理を行う。
* 以上の処理を行なうことで解くべき問題を多値分類から二値分類に変換することができた。
* ただ、ここまではまだ問題が全て解決できていない。
* これまでの処理では正例(正しい答え)についての学習に関する処理しか行なえておらず、負例(誤った答え)についての学習に関する処理を決める必要がある。
* 例えば、コンテキストが「you」と「goodbye」で正解となるターゲットが「say」の場合はこれまでの変換により、二値分類を行うことができるようになった。
* これは「you」と「goodbye」がコンテキストとして与えられた場合の正解が「say」であるという正例の場合である。
* つまり、重みが良い状態であれば、Sigmoidレイヤの出力として1に近い値を得ることができることを意味する。
* ここまでのニューラルネットワークでは上記のような正例の場合の推論についてだけ学習を行うことになる。
* 一方、負例の場合(推論時に「say」以外の単語を与えた場合)については学習することができず、高い精度の結果を返すことができない。
* 正例ではSigmoidレイヤの出力が1に近い値となり、負例ではSigmoidレイヤの出力が0に近い値になるようにする必要がある。
* つまり、コンテキストが「you」と「goodbye」のとき、ターゲットが「hello」である確率は低い値であることが望ましい。
* 多値分類問題を二値分類問題として扱うためには、正しい答えと間違った答えのそれぞれに対して正しく二値分類できる必要があるため、正例・負例の両方を対象として問題を考える必要がある(両方の場合の精度を上げる必要がある)。
* ただ全ての負例を対象として二値分類の学習を行なうことはしない。全ての負例を対象にしてしまうと語彙数が増えると計算量が増加してしまうためである。
* そこで、近似解として負例をいくつかピックアップし、それを少数サンプリングとして用いるようにする。
* この負例の少数サンプリングをNegative Samplingという手法となる。
* Negative Samplingという手法では、まず正例をターゲットとした場合の損失を求め、同時に負例をいくつかサンプリングし、その負例に関しても同様に損失を求める。
* そして、正例をターゲットにした場合の損失と負例をターゲットにした場合の損失を足し合わせ、その結果を最終的な損失とする手法である。
* 具体的な例として、コンテキストが「you」と「goodbye」のときの正例を「say」、負例としてサンプリングされた単語を「hello」と「I」とする。
* この場合のCBOWモデルの中間層以降の計算グラフについて考える。
* 正例(say)の場合はこれまでと同様、Sigmoid with Lossレイヤに正解ラベルとして「1」を入力し、負例(hello)の場合は、Sigmoid with Lossレイヤに正解ラベルとして「0」を入力するようにする。
* 最後にそれぞれのデータにおける損失を加算し、最終的な損失を出力する。
* 次に負例をどのようにサンプリングするかを考える。
* ここではコーパスの統計データに基づいてサンプリングを行うことを考える。この方法はランダムにサンプリングするよりもよい方法であるとして知られている。
* 具体的にはコーパスの中でよく使われる単語を抽出されやすくし、あまり使われない単語を抽出されにくくすることを意味する。
* コーパス内での単語の使用頻度に基づいて単語をサンプリングするには、まずコーパスから各単語の出現回数を求め、これを確率分布で表す。
* そしてその確率分布から単語をサンプリングする。
* 確率分布に従ってサンプリングすることで、コーパス内で多く登場した単語が抽出されやすくなり、逆にレアな単語は抽出されにくくなる。
* Negative Samplingでは、負例として多くの単語をカバーすることが望まれるが、計算量の問題から負例を少数に限定する必要がある。
* ここで負例としてレアな単語ばかりが選ばれてしまうと現実的な問題においてもレアな単語はほとんど出現しないため、よい結果が得られないことになる。
* つまり、レアな単語を相手にする重要度は低く、それよりも高頻出の単語に対応できるのが良い結果に繋がると言える。
* 以下では確率分布に従ってPythonを使い、単語をサンプリングする。ここでは、Numpyのnp.random.choice()メソッドを使用する。
   ```python
   >>> import numpy as np

   # 0～9からrandomに数字をサンプリング
   >>> np.random.choice(10)
   7
   >>> np.random.choice(10)
   2

   # wordsからrandomに要素をサンプリング
   >>> words = ['you', 'say', 'goodbye', 'I', 'hello', ',']
   >>> np.random.choice(words)
   'goodbye'

   # randomに5つの要素をサンプリング(重複あり)
   >>> np.random.choice(words, size=5)
   array(['goodbye', ',', 'hello', 'goodbye', 'say'], dtype='<U7')
   
   # randomに5つの要素をサンプリング(重複なし)
   >>> np.random.choice(words, size=5, replace=False)
   array(['hello', ',', 'goodbye', 'I', 'you'], dtype='<U7')

   # 確率分布に従ってサンプリング
   >>> p = [0.5, 0.1, 0.05, 0.2, 0.05, 0.1]
   >>> np.random.choice(words, p=p)
   'you'
   ```
* np.random.choice()の引数にsizeを指定することで、複数回サンプリングをまとめて行うことができ、また引数replaceにFalseを指定することでそれらのサンプリングを重複なく行なうことができる。
* また、引数pに確率分布を示すリストを指定することで、その確率分布に従ったサンプリングを行なうことができる。
* word2vecで提案されているNegative samplingでは以下のようになっている。
    * P'(wi) = P(wi)^0.75 / ΣP(wj)^0.75　　(4.4)
        * P(wi)：i番目の単語の確率
        * ΣP(wj) (j=1,2,...n)：P(w1)^0.75, P(w2)^0.75,...P(wn)^0.75の総和 = 1
* P'(wi)では通常の確率分布の各要素を0.75乗している。それに伴い、分母として0.75乗した各要素の確率分布の総和が必要となる。これは0.75乗後も確率の総和を１にするためである。
* 各要素を0.75乗するのは、出現確率の低い単語を見捨てないために行われている。
* つまり、0.75乗することで確率の低い単語の確率を少しだけ高くしている。
* 具体的には以下のようになる。
    ```python
    >>> p = [0.7, 0.29, 0.01]
    >>> new_p = np.power(p, 0.75)
    >>> new_p /= np.sum(new_p)
    >>> print(new_p)
    [0.64196878  0.33150408  0.02652714]
    ```
* 上記から変換前のある要素の確率が0.01だったものが、変換後は0.026...になっていることがわかる。
* また、0.75という数値に理論的な意味はなく、別の値を設定することも可能である。
* これらを踏まえ、コーパスから確率分布を作成後、各要素の確率を0.75乗し、np.random.choice()で負例をサンプリングする処理をUnigramSamplerクラスを実装し、使用する。
* Unigramとは、ひとつの連続した単語を意味し、UnigramSamplerクラスは１つの単語を対象に確率分布を作る。
* UnigramSamplerクラスは初期化時に以下の引数を取る。
    * corpus：単語IDリスト
    * power：確率分布に対する累乗の値(デフォルト値=0.75)
    * sample_size：負例サンプリングの個数
* またUnigramSamplerクラスはget_negative_sample()メソッドを持ち、引数としてtarget(正例の単語IDリスト)を取る。
* UnigramSamplerクラスを実際に使用する具体例は以下のようになる。
    ```python
    corpus = np.array([0, 1, 2, 3, 4, 1, 2, 3])
    power = 0.75
    sample_size = 2

    sampler = UnigramSampler(corpus, power, sample_size)
    target = np.array([1, 3, 0])
    negative_sample = sample.get_negative_sample(target)
    print(negative_sample)
    # [[0 3]
    #  [1 2]
    #  [2 3]]
    ```
* 上記では正例として、[1, 3, 0]をミニバッチとしており、それぞれのデータに対する負例を2つずつサンプリングしている。
* 1つ目のデータ1に対して、負例は[0 3]、2つ目のデータ3に対して、負例は[1 2]、3つ目のデータ0に対して、負例は[2 3]が生成されていることがわかる。
* 以下ではNegative Samplingを実装する。ここではNegativeSamplingLossというクラスで実装する。
* まず、コンストラクタの実装は以下のようになる。
    ```python
    class NegativeSamplingLoss:
        def __init__(sekf, W, corpus, power=0.75, sample_size=5):
            self.sample_size = sample_size
            self.sampler = UnigramSampler(corpus, power, sample_size)
            self.loss_layers = [SigmoidWithLoss() for _ in range(sample_size+1)]
            self.embed_dot_layers = [EmbeddingDot(W) for _ in range(sample_size+1)]

            self.params, self.grads = [], []
            for layer in self.embed_dot_layers:
                self.params += layer.params
                self.grads += layer.grads
    ```
* 初期化の引数はそれぞれ以下となる。
    * W：出力側の重み
    * corpus：コーパス(単語IDのリスト)
    * power確率分布の累乗の値
    * sample_size：負例のサンプリング数
* コンストラクタでは、まずUnigramSamplerクラスを生成し、そのインスタンスをメンバー変数samplerとして保持し、負例のサンプル数もメンバー変数sample_sizeに保持する。
* メンバー変数loss_layers、embed_dot_layersにはそれぞれ、SigmoidWithLoss()、EmbeddingDot()で生成されるレイヤを格納する。
* レイヤの数(各リストの要素する数)はsample+1個としており、これは正例用のレイヤを1つ、負例用のレイヤをsample_size個分生成することを意味する。
* ここでは、リストの先頭のレイヤが正例を扱うレイヤとする。つまり、loss_layers[0]、embed_dot_layers[0]が正例を扱うレイヤとなる。
* 最後に使用するパラメータと勾配をそれぞれ、params、gradsにリストとしてまとめている。
* 次に順伝播の実装は以下のようになる。
    ```python
        def forward(self, h, target)
            batch_size = target.shape[0]
            negative_sample = self.sampler.get_negative_sample(target)

            # 正例のforward処理
            score = self.embed_dot_layers[0].forward(h, target)
            correct_label = np.ones(batch_size, dtype=np.int32)
            loss = self.loss_layers[0].forward(score, correct_label)

            # 負例のforward処理
            negative_label = np.zeros(batch_size, dtype=np.int32)
            for i in range(self.sample_size):
                negative_target = negative_sample[:, i]
                score = self.embed_dot_layers[1+i].forward(h, negative_target)
                loss += self.loss_layers[1+i].forward(score, negative_label)
            
            return loss
    ```
* 順伝播処理を行なうforwardメソッドは以下の引数を取る。
    * h：中間層のニューロン
    * target：正例のターゲット
* 順伝播処理ではまず、self.samplerインスタンスのget_negative_sampleメソッドにより負例のサンプリングを行い、生成した負例をnegative_sampleに格納する。
* 以降では正例と負例それぞれのデータに対し、順伝播を行い、損失を加算していく。
* いずれもEmbeddingDotクラスのforwardメソッドでスコアを算出し、そのスコアとラベルをSigmoidWithLossクラスのforwardメソッドに渡して損失を求めている。
* ここでラベルは正例の場合は1、負例の場合は0をNumpyのones、zerosメソッドで生成して使用している。
* また、負例のforward処理は生成したsample_size分スコアの算出と損失の計算を行う。
* 最後に逆伝播の実装は以下のようになる。
    ```python
        def backward(self, dout=1):
            dh = 0
            for l0, l1 in zip(self.loss_layers, self.embed_dot_layers):
                dscore = l0.backward(dout)
                dh += l1.backward(dscore)
            
            return dh
    ```
* 逆伝播の実装では順伝播のときとは逆順に各レイヤのbackwardメソッドを実行する。
* また中間層のニューロンhは順伝播の差異に複数コピーされているため、その逆伝播では複数の勾配を加算することになる。

# 改良版word2vecの学習・評価
* これまでのEmbeddingレイヤ及びNegative Samplingという手法を実装してきた。
* 以下ではこれらの改良点を取り入れたニューラルネットワークを実装する。
* ここでは、PTBデータセットを使って学習し、より実用的な単語の分散表現獲得を目指す。
* CBOWモデルの実装はこれまでのSimpleCROWクラスにEmbeddingレイヤとNegative Sampling Lossレイヤを導入し、改良することで実現する。
* CBOWモデルの実装は以下の通り。
    ```python
    import sys
    sys.path.append('..')
    import numpy as np
    from common.layers import Embedding
    from ch04.negative_sampling_layer import NegativeSamplingLoss

    class CBOW:
        def __init__(self, vocab_size, hidden_size, window_size, corpus):
            V, H = vocab_size, hidden_size

            # Initialize weights
            W_in = 0.01 * np.random.randn(V, H).astype('f')
            W_out = 0.01 * np.random.randn(V, H).astype('f')

            # Create layers
            self.in_layers = []
            for i in range(2 * window_size):
                layer = EMbedding(W_in)
                self.in_layers.append(layer)
            self.ns_loss = NewgativeSamplingLoss(W_out, corpus, power=0.75, sample_size=5)

            # Create lists of all weight and gradient
            layers = self.in_layers + [self.ns_loss]
            self.params, self.grads = [], []
            for layer in layers:
                self.params += layer.params
                self.grads += layer.grads
            
            # Set the distributed expression of words to member variables.
            self.word_vecs = W_in
    ```
* 引数としてはSimpleCBOWクラスと同様、語彙数vocab_sizeと中間層のニューロンの数hidden_sizeを受け取る。
* 加えて、CBOWクラスでは単語IDのリストcorpus、コンテキストサイズ(周囲の単語をどれだけコンテキストとして含めるか)window_sizeを引数として受け取る。
* 重みの初期値はSimpleCBOWと同様、np.random.randn()を使用して、W_in、W_outを生成する。
* W_in、W_outはSimpleCBOWクラスでは形状が異なっている一方、CBOWクラスでは同じ形状となっている。
* これは、SimpleCBOWクラスでは出力側の重みが列方向に単語ベクトルが配置されている一方、CBOWクラスでは行方向に単語ベクトルを配置しているためである。
* これはNegativeSamplingLossクラス内でEmbeddingレイヤを使用しているためである。
* 次にレイヤを生成する。レイヤはwindow_size*2個分を生成する。
* 生成したレイヤはメンバー変数in_layersぶ配列としてまとめ、保持する。
* 加えて最後にNewgativeSamplingLossレイヤを生成し、同様にメンバー変数ns_lossに保持する。
* 生成したレイヤはlayersにまとめ、各パラメータと勾配をメンバー変数params、gradsにまとめ直す。
* また、後から単語の分散表現にアクセスできるよう、W_in(単語の分散表現)をメンバーword_vecsに格納する。
* 次に順伝播forward()、逆伝播backward()の実装は以下のようになる。
    ```python
        def forward(self, contexts, target):
            h = 0
            for i, layer in enumerate(self.in_layers):
                h += layer.forward(contexts[:,i])
            h *= 1 / len(self.in_layers)
            loss = self.ns_loss.forward(h, target)
            return loss

        def backward(self, dout=1):
            dout = self.ns_loss.backward(dout)
            dout *= 1 / len(self.in_layers)
            for layer in self.in_layers:
                layer.backward(dout)
            return None
    ```
* forward()メソッドの引数はSimpleCBOWクラスと同様、contextsとtargetの2つを取るが、SimpleCBOWではそれぞれone-hotベクトルだったのに対し、CBOWクラスでは単語IDとして扱われる。
* contexts、targetはSimpleCBOWクラスでは、以下の通りであった。
    * contexts：3次元NumPy配列：(コンテキストペアの数,コンテキストサイズ,語彙数)
    * target：2次元NumPy配列：(コンテキストペアの数,語彙数)
* 一方、CBOWクラスにおけるcontexts、targetは以下の通りとなる。
    * contexts：単語IDを要素とする2次元NumPy配列
    * target：単語IDを要素とする1次元NumPy配列
* 以上を踏まえ、CBOWモデルを使用した学習の実装は以下の通りとなる。
    ```python
    import sys
    sys.path.append('..')
    import numpy as np
    from common import config
    # config.GPU = True　# GPUを使用する場合はコメントを外す
    import pickle
    from common.trainer import Trainer
    from common.optimizer import Adam
    from cbow import CBOW
    from common.util import create_contexts_target, to_cpu, to_gpu
    from dataset import ptb

    # Set hyper-parameter
    window_size = 5
    hidden_size = 100
    batch_size = 100
    max_epoch = 10

    # Load data set
    corpus, word_to_id, id_to_word = ptb.load_data('train')
    vocab_size = len(word_to_id)

    contexts, target = create_contexts_target(corpus, window_size)
    if config.GPU:
        contexts, target = to_gpu(contexts), to_gpu(target)
    
    # Create model etc...
    model = CBOW(vocal_size, hidden_size, window_size, corpus)
    optimizer = Adam()
    trainer = Trainer(model, optimizer)

    # Start learning
    trainer.fit(contexts, target, max_epoch, batch_size)
    trainer.plot()

    # Set the various necessary data to member variables.
    word_vecs = model.word_vecs
    if config.GPU:
        word_vecs = to_cpu(word_vecs)
    params = {}
    params['word_vecs'] = word_vecs.astype(np.float16)
    params['word_to_id'] = word_to_id
    params['id_to_word'] = id_to_word
    pkl_file = 'cbow_params.pkl'
    with open(pkl_file, 'wb') as f:
        pickle.dump(params, f, -1)
    ```
* 上記の実装では以下のようにハイパーパラメータを固定している。
    * ウィンドウサイズ：5
    * 隠れ層のニューロン数：100
* 対象のコーパスにもよるが、一般的にウィンドウサイズは2～10、中間層のニューロン数(=単語の分散表現の次元数)は50～100くらいに設定すると良い結果が得られることが多い。
* また、上記の実装で使用しているPTBコーパスはこれまでに使用してきたコーパス'You say goodbye and I say hello.'に比べて格段にサイズが大きい。
* そのため、学習には多くの時間がかかり、通常のCPUでは半日程度かかる。
* ここではオプションとしてGPUを使って学習を行なえるモードを用意している。
    * 上記の実装でコメントアウトしているconfig.GPU = True部分を有効にすることでGPUを使った学習を行うことができる。
* ただし、GPUで学習を行うためには、学習環境でNVIDIAのGPUを備えていてCuPyがインストールされている必要がある。
* また、fit()で学習を行ったら、重み(ここでは入力側の重み)を取り出し、ファイルに保存する。
* ここではPythonコード中のオブジェクトをファイルに保存するためにpickleを使用している。
* 次にCBOWモデルを使用した学習で作成したモデルの評価を行う。
* モデルの評価の実装は以下の通りとなる。
    ```python
    import sys
    sys.path.append('..')
    from common.util import most_similar
    import pickle

    pkl_file = 'cbow_params.pkl'

    with open(pkl_file, 'wb') as f:
        params = pickle.load(f)
        word_vecs = params['word_vecs']
        word_to_id = params['word_to_id']
        id_to_word = params['id_to_word']
    
    querys = ['you', 'year', 'car', 'toyota']
    for query in querys:
        most_similar(query, word_to_id, id_to_word, word_vecs, top=5)
    ```
* まず、学習の結果得られた重みを保存したファイルを開き、各要素を取り出し、変数に格納する。
* それらと評価用データをmost_similar()に渡す。
* most_similar()により、querysの各単語を与えることで、その単語に対する類似単語を上位から順に表示することができる。
* この実装を実行すると、以下のような結果が得られる。
    ```
    [query] you
        we:        0.610597074032
        someone:   0.591710150242
        i:         0.554366409779
        something: 0.490028560162
        anyone:    0.473472118378
    
    [query] year
        month:  0.718261063099
        week:   0.652263045311
        spring: 0.62699586153
        summer: 0.625829637051
        decade: 0.603022158146
    
    [query] car
        luxury:     0.497202396393
        arabia:     0.478033810854
        auto:       0.471043765545
        disk-drive: 0.450782179832
        travel:     0.40902107954
    
    [query] toyota
        ford:            0.550541639328
        instrumentation: 0.510020911694
        mazda:           0.49361255765
        bethlehem:       0.474817842245
        nissan:          0.474622786045
    ```
* 上記の結果から、queryに対して、類似の単語が得られていることがわかる。
* つまり、CBOWモデルで獲得した単語の分散表現の性質が良いものであると言える。
* また、word2vecで得られた単語の分散表現は、類似単語を集めるだけでなく、より複雑なパターンを捉えることができることがわかっている。
* 例としては、「king - man + woman = queen」といった類推問題が挙げられる。
* これはword2vecの単語の分散表現を用いることで類推問題をベクトルの加算と減算で解くことができること意味している。
* 実際にこのような類推問題を解くのは、「man : woman = king : ?」の"?"を類推することになり、これは単語ベクトル空間上のman→womanというベクトルに対し、king→"?"というベクトルができるだけ近くなる単語"?"を探すことを意味する。
* ここで、単語"man"の分散表現(単語ベクトル)を"vec('man')"で表すとすると、「man→womanというベクトルに対し、king→"?"というベクトルができるだけ近くなる単語"?"を探す」ことは、以下の関係性で表すことができる。
    * vec('woman') - vec('man') = vec('?') - vec('king')
* これを変形すると、以下のようになる。
    * vec('?') = vec('king') + vec('woman') - vec('man')
* 上記より、"?"を類推する問題は、vec('king') + vec('woman') - vec('man')を求める問題への帰着することができる。
* この問題を解く関数をanalogy()として定義する。
* analogy()は以下のように引数として情報を渡すことでvec('?')を求めることができる。
    * analogy('man', 'king', 'woman', word_to_id, id_to_word, word_vecs, top=5)
* またanalogy()は以下のようにスコアが高い順に5つの単語が出力されるものとする。
    ```
    [analogy] man:king = woman:?
        word1: 5.003233
        word2: 4.400302
        word3: 4.22342
        word4: 4.003234
        word5: 3.934550
    ```
* これを踏まえ、以下のような引数を指定して、analogy()関数を実行する。
    * analogy('king', 'man', 'queen', word_to_id, id_to_word, word_vecs, top=5)
    * analogy('take', 'took', 'go', word_to_id, id_to_word, word_vecs, top=5)
    * analogy('car', 'cars', 'child', word_to_id, id_to_word, word_vecs, top=5)
    * analogy('good', 'better', 'bad', word_to_id, id_to_word, word_vecs, top=5)
* 上記を実行すると、それぞれ以下の結果が得られる。
    ```
    [analogy] king:man = queen:?
        woman: 5.161407947540283
        veto: 4.928170680999756
        ounce: 4.689689636230469
        earthquake: 4.633471488952637
        successor: 4.6089653968811035
    
    [analogy] take:took = go:?
        went: 4.548568248748779
        points: 4.248863220214844
        began: 4.090967178344727
        comes: 3.9805688858032227
        oct: 3.9044761657714844

    [analogy] car:cars = child:?
        children: 5.217921257019043
        average: 4.725458145141602
        yield: 4.208011627197266
        cattle: 4.18687629699707
        priced: 4.178797245025635

    [analogy] good:better = bad:?
        more: 6.647829532623291
        less: 6.063825607299805
        rather: 5.220577716827393
        slower: 4.733833312988281
        greater: 4.672840118408203
    ```
* この結果は概ね期待した通りの結果と言える。
* 1～3番目は正しく答えられていると言える一方、4番目は正しく答えられていない。
* 与えられた単語の比較級を答えるべきだが、"more"となってしまい、正しい答えは"worse"である。
* ただ上位に挙げられているのは、"more"、"less"などの比較級の単語となっており、正しくはないもののある程度の性質は単語の分散表現にエンコードできていると言える。
* このようにword2vecでは得られた単語の分散表現を使うことでベクトルの加減算により、単語の類推問題を解くことができる。
* さらに単語の意味だけではなく、文法的な情報も捉えることができていることがわかる。
* ただし、PTBデータセットは単語の類推問題を解くには小規模であり、多くの類推問題を正しく解くことができない。
* 類推問題を精度良く解くためには、より大きなコーパスを対象として学習を行う必要がある。

# word2vecのアプリケーションの例
* word2vecで得られた単語の分散表現は類似単語を求めることができる、という利点に加え、転移学習(transfer learning)に利用できることが重要な利点として挙げられる。
* 転移学習により、ある分野で学習した知識を他の分野にも適用することができる。
* 自然言語のタスクを解く場合、通常word2vecによる単語の分散表現をゼロから学習することはほとんどない。
* 多くの場合、大きなコーパス(Wikipedia, Google Newsのテキストデータなど)で学習済みの単語の分散表現を個別のタスクに利用する。
* テキスト分類や文書クラスクラスタリング、品詞タグ付け、感情分析といったタスクにおいて、最初のステップでは単語をベクトルに変換する必要がある。
* このときに学習済みの単語の分散表現を利用する。
* また単語の分散表現の利点は、単語を固定長のベクトルに変換できることにある。
* さらに文章(単語の並び)に対しても、単語の分散表現を使うことで固定長のベクトルに変換でき、文章をどのように固定長のベクトルに変換するかは多く研究されている。
* 最も単純な方法は文章の各単語の分散表現の総和を求める方法が考えられる。
* これはbag-of-wordsと呼ばれ、単語の順序を考慮しないモデルである。
* また、リカレントニューラルネットワーク(RNN)を使うことでword2vecの単語の分散表現を利用しつつ、文章を固定長のベクトルに変換することができる。
* 単語や文章を固定長のベクトルに変換できることは、一般的な機械学習の手法(ニューラルネットワークやSVMなど)が適用できることを意味するため、非常に重要である。
* このことは以下のような処理の流れ(パイプライン)で表すことができる。
    * 質問(自然言語) → 単語のベクトル化(word2vec) → 機械学習システム(ニューラルネットワークやSVMなど) → 答え
* ベクトル化された固定長のベクトルは機械学習システムへの入力となる。つまり、機械学習システムによって、目的の答えを出力することができることを意味する。
* このようなパイプラインにおいては通常、単語の分散表現の学習と機械学習システムの学習は別のデータセットを使って個別に学習を行なう。
    * 単語の分散表現の学習はWikipediaのような汎用的なコーパスを使って学習を先に済ませておく。
    * 一方、現状のタスクに関してはそのタスクのために集められたデータを対象に機械学習システムに入力して学習を行う。
* 以下では具体敵に単語の分散表現の使い方を見ていく。
* ここでは利用者が1億人を超えるスマートフォンアプリの開発・運営をしているとする。
* そのアプリに関して、ユーザからは多くの意見やつぶやきが届けられる。それらの中には好意的なものもあれば、不満を持っていると考えられるものもある。
* そこで、これらのユーザの声を自動で分類するシステムを作ることを考える。
* 例えば、ユーザから送られてくるメールの内容からユーザの感情を3段階に分類する。
* これにより、不満を持つユーザの声にいち早く目を通すことを実現する。
* これはアプリの致命的な問題を発見し、早期に手を打つことでユーザの満足度を改善することができること意味する。
* メールの自動分類システムを作成するためには、まずメール(データ)を収集する必要がある。
* 集めたメールはそれぞれに対して、人手でラベル付けを行う。つまり3段階の感情を表すラベル(positive/neutral/negative)を付与する。
* ラベル付けを行なったら、学習済みのword2vecを用いて、メールの文章をベクトルに変換します。
* 最後に感情分析を行う何らかの分類システムに対して、ベクトルかされたメールと感情ラベルを与えて学習する。

# 単語ベクトルの評価方法
* word2vecで得られた単語の分散表現の良さはどう評価すべきかを考える。
* 単語の分散表現は上述のような感情分析のように現実的には何らかのアプリケーションで使われることが多い。
* その場合、望まれるのは精度が良いシステムであるが、そのシステムは複数のシステムで構成されていることを考えておく必要がある。
* 例えば、上述の感情分析を行うシステムでは、単語の分散表現を作るシステムと特定の問題に対して分類を行うシステムがある。
* つまり感情分析を行うシステムは2段階の学習を行なった上で評価する必要があり、2つのシステムにおいて最適なハイパーパラメータのためのチューニングも必要となる。
* そこで、単語の分散表現の良さのみを評価する場合は、現実的なアプリケーションとは切り離して評価するのが一般的である。
* その際によく用いられる単語の分散表現の評価指標は単語の類似性や類推問題による評価である。
* まず単語の類似性の評価では、人間が作成した単語類似度の評価セットを使って評価が行われる。
* 例えば、0～10でスコア化するとすると、"animal"と"cat"は8、"cat"と"car"は2というように人が単語間の類似性を採点する。
* そのスコアとword2vecによるコサイン類似度のスコアを比較して相関性を確認する。
* 次に類推問題による評価は「king : queen = man : ?」のような類推問題を出題し、その正解率で単語の分散表現を評価する。
* ある論文ではword2vecのモデル、単語の分散表現の次元数、学習したコーパスのサイズをパラメータとして比較実験を行っている。
* これらに対し、Semantics(単語の意味を類推する問題)、Syntax(単語の形状情報を問う問題)の問題を解かせ、その正答率を比較する。
* この比較実験から分かることは以下の通り。
    * モデルによって精度が異なるため、コーパスに応じて最適なモデルを選ぶ必要がある。
    * 学習に使用したコーパスが大きいほど、良い結果が得られる。
    * 単語ベクトルの次元数は適度な大きさが必要で大きすぎても小さすぎても精度が悪くなる。
* 類推問題を解いた結果を比較によって、そのモデルが単語の意味や文法的な問題を正しく理解しているかを計測することができる。
* また、類推問題を精度良く説くことができるモデルであれば、自然言語を扱うアプリケーションでも良い結果を得られることが期待できる。
* ただし、モデルの良さ(単語の分散表現の良さ)がアプリケーションにどれだけ貢献できるかはアプリケーションの種類やコーパスの内容など取り扱う問題の状況に応じて変化する。
* つまり、類推問題を精度良く解けたとしても、アプリケーションで良い結果が得られるとは限らない。

# リカレントニューラルネットワーク(RNN)
* a
* ★★～P.172★★




# Method
* P.64：text.lower()
* P.64：text.replace('.', '. ')
* P.64：text,split(' ')
* P.66：[word_to_id[w] for w in words]　※内包表記
* P.69：sys.path.append('..')
* P.72：enumerate(corpus)
* P.75：if query not in word_to_id:
* P.75：x.argsort()
* P.80：np.set_printoptions(precision=3)
* P.85：word_to_id.items()
* P.85：plot.annotate(word, (U[word_id, 0], U[word_id, 1]))
* P.85：plt.scatter(U[:,0], U[:,1], alpha=0.5)
* P.99：np.random.randn(7, 3)
* P.99：np.dot(c, W)
* P.113：range(start, stop)　※start～stop-1までの整数を生成
* P.138：dW[...] = 0
* P.157：self.loss_layers = [SigmoidWithLoss() for _ in range(sample_size+1)]　※内包表記
* P.159：zip(self.loss_layers, self.embed_dot_layers)



