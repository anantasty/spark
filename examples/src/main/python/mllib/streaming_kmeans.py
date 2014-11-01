#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""
A K-means streaming example

"""

import sys

from pyspark import SparkContext
from pyspark.streaming.context import StreamingContext

from pyspark.mllib import regression

USAGE = "streaming_kmeans.py train_dir predict_dir"

if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.exit(USAGE)

    sc = SparkContext("Streaming KMeans")
    ssc = StreamingContext(sc, 5)
    train_data = ssc.textFileStream("/home/anant/projects/train_data").map(
        lambda row: regression.LabeledPoint.parse_point(row))
    test_data = ssc.textFileStream("/home/anant/projects/test_data").map(
        lambda row: regression.LabeledPoint.parse_point(row).features)
    model = regression.StreamingLinearRegressionWithSGD(
        ssc, initialWeights = [0.0, 0.0] * 2)
    model.trainOn(train_data)
    model.predictOn(test_data)

    ssc.start()
    ssc.awaitTermination()
