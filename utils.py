import json
import numpy as np
import matplotlib.pyplot as plt
from utils import *

# Constantes globais.
VALID_PROJECTS = ['dotnet/roslyn','ppy/osu','Particular/NServiceBus','Humanizr/Humanizer','npgsql/npgsql','gui-cs/Terminal.Gui','projectkudu/kudu','vkhorikov/CSharpFunctionalExtensions','dotnet/BenchmarkDotNet','ninject/Ninject','FlaUI/FlaUI','kiegroup/jbpm','gwtproject/gwt','ctripcorp/x-pipe','geogebra/geogebra','spring-projects/spring-batch','opennetworkinglab/onos','kiegroup/optaplanner','treasure-data/digdag','apache/parquet-mr','cache2k/cache2k','cucumber/cucumber-jvm','hub4j/github-api','manifold-systems/manifold','leangen/graphql-spqr','tony-framework/TonY','briandilley/jsonrpc4j','hierynomus/sshj','dlew/joda-time-android','scobal/seyren','mesosphere/marathon-ui','kriskowal/q-io','nitoyon/livereloadx','laravel/framework','ThinkUpLLC/ThinkUp','wp-cli/wp-cli','ray-project/ray','RasaHQ/rasa','dask/dask','iterative/dvc','binux/pyspider','trentm/python-markdown2','manrajgrover/halo','apache/lucenenet','lra/mackup','nhibernate/nhibernate-core','NancyFx/Nancy','mcxiaoke/android-volley','yiisoft/yii2','httpie/http-prompt','dotnet/reactive','thombergs/code-examples','akkadotnet/akka.net','zkSNACKs/WalletWasabi','rabbitmq/rabbitmq-dotnet-client','aaubry/YamlDotNet','naudio/NAudio','git-tfs/git-tfs','tmenier/Flurl','apache/phoenix','locationtech/jts','SpartnerNL/Laravel-Excel','phpspec/phpspec','PennyLaneAI/pennylane','mono/monodevelop','getsentry/sentry-php','xunit/xunit','wix/react-native-calendars','AutoFixture/AutoFixture','FluentValidation/FluentValidation','plotly/dash','protobuf-net/protobuf-net','Lidarr/Lidarr','dotnet/Open-XML-SDK','find-sec-bugs/find-sec-bugs','fluentassertions/fluentassertions','VFK/gulp-html-replace','SignalR/SignalR','mher/node-celery','captbaritone/raven-for-redux','castleproject/Core','apache/kylin','openiddict/openiddict-core','bobthecow/mustache.php','webiny/webiny-js','IdentityServer/IdentityServer3','filp/whoops','icsharpcode/ILSpy','igniterealtime/Smack','nsubstitute/NSubstitute','tdebatty/java-string-similarity','sebastianbergmann/phpunit-mock-objects','saltstack/salt','encode/django-rest-framework','AppMetrics/AppMetrics','abel533/ECharts','hardkoded/puppeteer-sharp','MahApps/MahApps.Metro','dmlc/dgl','Unity-Technologies/ml-agents','owncloud/core','thephpleague/csv','Cysharp/MagicOnion','Ocramius/ProxyManager','zeromq/netmq','docker-java/docker-java','MessagePack-CSharp/MessagePack-CSharp','akarnokd/RxJavaInterop','mRemoteNG/mRemoteNG','symfony/security-core','bitwarden/server','picoe/Eto','OWASP/SecurityShepherd','Bukkit/Bukkit','nitaliano/react-native-mapbox-gl','walmartlabs/lumbar','bitovi/documentjs','biggora/caminte','zephir-lang/zephir','doctrine/migrations','symfony/doctrine-bridge','symfony/dom-crawler','composer/satis','mqtt-tools/mqttwarn','katspaugh/wavesurfer.js','DapperLib/Dapper','RicoSuter/NSwag','p-org/P','PowerShell/PSReadLine','r0adkll/Slidr','awslabs/lambda-streams-to-firehose','cakephp/phinx','symfony/http-foundation','guzzle/psr7','schmittjoh/JMSSerializerBundle','symfony/process','thephpleague/oauth2-client','symfony/finder','KnpLabs/KnpPaginatorBundle','frappe/frappe_docker','microsoft/Git-Credential-Manager-for-Windows','Rajawali/Rajawali','dlazaro66/QRCodeReaderView','barryvdh/laravel-ide-helper','prowler-cloud/prowler','lepture/python-livereload','kynan/nbstripout','apollographql/apollo-client','GeekyAnts/NativeBase','neuecc/UniRx','SaltwaterC/aws2js','apiaryio/curl-trace-parser','sindresorhus/grunt-recess','Payum/Payum','blueimp/jQuery-File-Upload','guzzle/guzzle','bolt/bolt','FriendsOfSymfony/FOSRestBundle','silexphp/Silex','beberlei/DoctrineExtensions','michelf/php-markdown','PyWavelets/pywt','jina-ai/clip-as-service','mfogel/django-timezone-field','jazzband/django-configurations','nonebot/nonebot','spadgos/sublime-jsdocs','kootenpv/yagmail','kulshekhar/ts-jest','Caliburn-Micro/Caliburn.Micro','PHP-CS-Fixer/PHP-CS-Fixer','Intervention/image','strapdata/elassandra','apache/incubator-heron','roboguice/roboguice','eclipse/jetty.project','huggingface/transformers','fastnlp/fastNLP','linkedin/rest.li','vladmihalcea/high-performance-java-persistence','php-telegram-bot/core','bcgit/bc-java','DozerMapper/dozer','phpspec/prophecy','numba/numba','google/jax','mypaint/mypaint','pyodide/pyodide','catalyst-team/catalyst','iterate-ch/cyberduck','ramsey/uuid','mongodb/mongo-java-driver','vyuldashev/laravel-queue-rabbitmq','spring-projects/spring-data-jpa','diversario/node-ssdp','gitblit-org/gitblit','adamfisk/LittleProxy','internetarchive/heritrix3','bootique/bootique','Netflix/Fenzo','wallabag/wallabag','drewnoakes/metadata-extractor','curran/model','digiaonline/react-foundation','elastic/eui','jenkinsci/configuration-as-code-plugin','webonyx/graphql-php','composer/composer','williamfiset/DEPRECATED-data-structures','VictorAlbertos/RxCache','classgraph/classgraph','projectmesa/mesa','f-droid/fdroidclient','lyft/scoop','defuse/php-encryption','ethereum/py-evm','mvantellingen/python-zeep','vladmihalcea/flexy-pool','pedrovgs/Renderers','kubeflow/kubeflow','proninyaroslav/libretorrent','awsdocs/aws-lambda-developer-guide','dustin10/VichUploaderBundle','open-mmlab/mmdetection','aws-ia/taskcat','Zulko/moviepy','odrotbohm/spring-restbucks','phonegap/phonegap-plugin-push','bluelinelabs/LoganSquare','symfony/framework-bundle','spatie/laravel-permission','sockeqwe/fragmentargs','symfony/routing','symfony/filesystem','Railk/T3S','serbanghita/Mobile-Detect','christianalfoni/flux-angular','getsentry/sentry-javascript','seek-oss/seek-style-guide','ilius/pyglossary','yuku/old-textcomplete','zabbix/zabbix','gaearon/react-proxy','pluralsight/react-styleable','baudehlo/node-phantom-simple','symfony/config','googlearchive/backbonefire','symfony/yaml','composer/semver','vercel/hyper','tensorflow/tensorboard','noopkat/electric-io','smogon/pokemon-showdown','artsy/ezel','maxmind/GeoIP2-python','PrestaShop/PrestaShop','dompdf/dompdf','phpmyadmin/phpmyadmin','facebookresearch/fairseq','symfony/dependency-injection','Respect/Validation','jenssegers/date','symfony/console','symfony/var-dumper','WordPress/Requests','python-pillow/Pillow','moneyphp/money','pyelasticsearch/pyelasticsearch','thephpleague/fractal','lstrojny/functional-php','codingo/VHostScan','doctrine/annotations','doctrine/common','symfony/browser-kit','orvice/ss-panel','Miserlou/Zappa','elastic/kibana']

def get_box_plot_data(labels, bp):
    """
    Extrai dados do box plot para um formato mais amigável.

    Args:
        labels: Lista de rótulos para cada box plot.
        bp: Dicionário retornado pela função `matplotlib.pyplot.boxplot`.

    Returns:
        Lista de dicionários, onde cada dicionário representa um box plot e contém
        as chaves: 'label', 'lower_whisker', 'lower_quartile', 'median',
        'upper_quartile', 'upper_whisker'.
    """
    rows_list = []
    for i, label in enumerate(labels):  # Usando enumerate para simplificar o loop
        rows_list.append({
            'label': label,
            'lower_whisker': bp['whiskers'][i*2].get_ydata()[1],
            'lower_quartile': bp['boxes'][i].get_ydata()[1],
            'median': bp['medians'][i].get_ydata()[1],
            'upper_quartile': bp['boxes'][i].get_ydata()[2],
            'upper_whisker': bp['whiskers'][i*2 + 1].get_ydata()[1]
        })
    return rows_list


def z_normalize(ts):
    """
    Aplica a normalização Z a uma série temporal.

    Args:
        ts: Série temporal (array 1D NumPy).

    Returns:
        Série temporal normalizada. Retorna a série original
        se o desvio padrão for zero para evitar divisão por zero.
    """
    mean_ts = np.mean(ts)
    std_ts = np.std(ts)
    return (ts - mean_ts) / std_ts if std_ts != 0 else ts - mean_ts


def sbd(ts1, ts2):
    """
    Calcula a Shape-Based Distance (SBD) entre duas séries temporais.

    Args:
        ts1: Primeira série temporal (array 1D NumPy).
        ts2: Segunda série temporal (array 1D NumPy).

    Returns:
        Uma tupla contendo:
            - distance: A SBD entre as duas séries temporais.
            - xcorr_offset: O deslocamento onde ocorre a melhor correlação.
    """
    ts1 = z_normalize(ts1)
    ts2 = z_normalize(ts2)

    xcorr = np.correlate(ts1, ts2, mode='full')
    xcorr_offset = np.argmax(np.abs(xcorr)) - (len(ts2) - 1)

    # Calcula o NCC (Normalized Cross-Correlation)
    NCC = xcorr.max() / (len(ts1) * np.std(ts1) * np.std(ts2)) if np.std(ts1) * np.std(ts2) !=0 else 0


    distance = 1 - NCC
    return distance, xcorr_offset


def process_similarity(X, labels, centroids, algorithm_name):
    """
    Processa e calcula a similaridade SBD para cada série temporal em relação ao seu centróide.

    Args:
        X: Conjunto de dados das séries temporais.
        labels: Rótulos de cluster para cada série temporal.
        centroids: Centróides dos clusters.
        algorithm_name: Nome do algoritmo de clustering usado (para fins de logging/exibição).

    Returns:
        Uma lista de valores SBD.
    """
    unique_labels = np.unique(labels)
    sbd_n = []
    for k in unique_labels:
        if k == -1:
            continue  # Ignorar o ruído

        class_member_mask = (labels == k)
        for series in X[class_member_mask]:
            distance, _ = sbd(series.flatten(), centroids[k].flatten())  # Ignorar o offset
            sbd_n.append(distance)
    return sbd_n



def normalize_ks_centroids(centroid):
    """
    Normaliza os valores do centróide para a faixa de 0 a 100.

    Args:
        centroid: Array NumPy com os valores do centróide.

    Returns:
       Array NumPy com os valores normalizados entre 0 e 100.
    """
    min_val = np.min(centroid)
    max_val = np.max(centroid)
    # Evitando divisão por zero. Se min_val == max_val, retorna 0
    normalized_centroid = (centroid - min_val) / (max_val - min_val) * 100 if (max_val- min_val) != 0 else np.zeros_like(centroid)
    return normalized_centroid



def plot_individual_clusters(X, labels, centroids, algorithm_name):
    """
    Plota cada cluster individualmente com suas séries temporais e centróide.

    Args:
        X: Conjunto de dados das séries temporais.
        labels: Rótulos de cluster para cada série temporal.
        centroids: Centróides dos clusters.
        algorithm_name:  Nome do algoritmo usado (para o título do gráfico).
    """

    unique_labels = np.unique(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

    for k, col in zip(unique_labels, colors):
        if k == -1:
            continue  # Ignorar o ruído

        class_member_mask = (labels == k)
        num_repos = np.sum(class_member_mask)

        print(f"Cluster {k} possui {num_repos} repositórios")

        for series in X[class_member_mask]:
            plt.plot(series.flatten(), '--', markerfacecolor=tuple(col), markeredgecolor='k', markersize=6)

        plt.plot(centroids[k].flatten(), 'k', color='black', markeredgecolor='k', markersize=12, linewidth=3)
        plt.title(f'{algorithm_name} - Cluster {k}')
        plt.xlabel('Time')
        plt.ylabel('Time series')
        plt.legend()
        plt.ylim(top=100)
        plt.show()


def load_data_from_json(json_file):
    """
    Carrega dados de séries temporais de um arquivo JSON.

    Args:
        json_file: Caminho para o arquivo JSON.

    Returns:
        Tupla contendo:
            - ids: Array NumPy com os IDs dos repositórios.
            - timeseries: Array NumPy com as séries temporais.
    """
    with open(json_file, 'r') as file:
        data = json.load(file)

    ids = [entry['id'] for entry in data if entry['full_name'] in VALID_PROJECTS]
    timeseries = [entry['timeseries'] for entry in data if entry['full_name'] in VALID_PROJECTS]
    return np.array(ids), np.array(timeseries)