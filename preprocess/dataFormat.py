def format(fromPath, toPath):
    zeroNum = 0
    oneNum = 0
    with open(fromPath, 'r') as f1:  # 读取模式读取待写入的文件
        with open(toPath, 'a') as f2:  # 追加模式保存格式化后的数据
            line = f1.readline()
            while line:
                if line[0] == '>':  # 如果是标记行则读取标记
                    category = line[-2]  # 倒数第二个字符是类别
                    if category == '0':
                        zeroNum += 1
                        category = '1'
                    else:
                        oneNum += 1
                        category = '0'
                    spacePos = line.find(' ')  # 找到空格的位置
                    name = line[:spacePos]
                    formattedLine = name + '|' + category + '|' + 'testing\n'
                    f2.write(formattedLine)
                else:  # 不是标记行则读取序列
                    seq = line
                    f2.write(seq)
                line = f1.readline()
    print("Label为0的个数：{}\tLabel为1的个数{}".format(zeroNum, oneNum))


format("ProCla/data/AA-index/S/test.txt", "ProCla/data/AA-index/S/test_formatted.txt")
