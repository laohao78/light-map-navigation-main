std_msgs/Header header
sensor_msgs/Image source_image          # 原始图像
sensor_msgs/Image label_image           # 整体标签图像(int16,每个像素值对应检测索引)
string[] class_names                    # 类别名称数组
float64[] confidences                   # 置信度数组
sensor_msgs/RegionOfInterest[] boxes    # 边界框数组