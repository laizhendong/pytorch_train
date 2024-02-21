import onnx

onnx_model=onnx.load("C:/Users/lzd/Desktop/yolov5_obb-master/pth2onnx/yolov5s_bansi_oriented.onnx")
graph=onnx_model.graph


node=graph.node
input = graph.input 

#print(node) 
print(input) 
print(graph.output) 

graph.output.remove(graph.output[1])
graph.output.remove(graph.output[1])
graph.output.remove(graph.output[1])

print(graph.output) 
#for i in range(len(node)):
    #print(node[i])


#搜索目标节点
'''
for i in range(len(node)):
    if node[i].name=="345":
        node_rise=node[i]   
        #if node_rise.output[0]=="2207":
            #print("node is :",i)
            #print("node_rise:",node_rise)



old_scale_node=node[1568]
new_scale_node=onnx.helper.make_node(
    "Clip",
    inputs=["2180","2524","2203"],
    outputs=['2207'],
    name="Clip_1568"
)

old_scale_node1=node[1574]
new_scale_node1=onnx.helper.make_node(
    "Clip",
    inputs=["2180","2525","2211"],
    outputs=['2215'],
    name="Clip_1574"
)
old_scale_node2= node[1567]
old_scale_node3= node[1573]


old_scale_node=node[1568]
graph.node.remove(old_scale_node)
#graph.node.insert(1568,new_scale_node)
#graph.node.remove(old_scale_node1)


for j in range(len(graph.node)):
    print("graph_node:", graph.node[j])
    if graph.node[j].op_type=="Clip_new":
        node_rise_new = graph.node[j]
        if node_rise_new.output[0]=="2207":
            print("node is :",i)
            print("change node_rise:",node_rise_new)
'''

onnx.checker.check_model(onnx_model)
onnx.save(onnx_model,"C:/Users/lzd/Desktop/yolov5_obb-master/pth2onnx/yolov5s_bansi_obb_old_refine.onnx")