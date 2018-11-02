import  json

file =  open('../data/results/test.predicted.json')
result = []
for line in file:
    line = line.strip()
    _json =  json.loads(line)
    _answer =  _json['pred_answer']
    _id = _json['query_id']
    result.append(str(_id)+'\t'+_answer)


result = '\n'.join(result)

file = open('../data/results/final.txt','w',encoding='utf-8')
file.write(result)
file.close()
