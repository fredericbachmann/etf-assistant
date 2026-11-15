# OpenSearch with only the plugins the app uses, so it starts faster and needs less memory:
#   opensearch-knn            vector field and k-NN queries
#   opensearch-neural-search  hybrid query and the normalization processor (search pipeline)
#   opensearch-ml             required by neural-search
#   opensearch-job-scheduler  required by opensearch-ml
# Security is removed too: the port is only reachable from the app's Docker network.
FROM opensearchproject/opensearch:3.8.0

# dependents first: a plugin can't be removed while another one extends it
RUN for p in \
      opensearch-skills opensearch-flow-framework opensearch-security-analytics opensearch-alerting \
      opensearch-anomaly-detection opensearch-index-management opensearch-reports-scheduler \
      opensearch-observability opensearch-notifications opensearch-notifications-core \
      opensearch-asynchronous-search opensearch-cross-cluster-replication opensearch-custom-codecs \
      opensearch-geospatial opensearch-ltr opensearch-performance-analyzer opensearch-search-relevance \
      opensearch-security opensearch-sql opensearch-system-templates opensearch-ubi query-insights; \
    do bin/opensearch-plugin remove "$p" || exit 1; done \
 && bin/opensearch-plugin list
