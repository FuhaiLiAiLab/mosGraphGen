library(shiny)
library(dplyr)
library(igraph)
library(networkD3)
library(kdensity)
library(ggplot2)
library(Jmisc)

round_df <- function(df, digits) {
  nums <- vapply(df, is.numeric, FUN.VALUE = logical(1))
  df[,nums] <- round(df[,nums], digits = digits)
  (df)
}

### 1. READ GRAPH [edge_index, node] FROM FILES
# setwd('C:/Users/hemingzhang/Documents/vs-files/OmicsGraph')
setwd('/Users/muhaha/Files/VS-Files/OmicsGraph')
net_edge_weight = read.csv('./UCSC-analysis/fold_1/fold_avg_edge_weight_df.csv')
colnames(net_edge_weight)[1] <- 'From'
colnames(net_edge_weight)[2] <- 'To' 
net_node = read.csv('./UCSC-graph-data/map-all-gene.csv') # NODE LABEL
  
### 2.1 FILTER EDGE BY [edge_weight]
edge_threshold = 0.5
filter_net_edge = filter(net_edge_weight, edge_weight_avg > edge_threshold)
filter_net_edge_node = unique(c(filter_net_edge$From, filter_net_edge$To))
filter_net_node = net_node[net_node$Gene_num %in% filter_net_edge_node, ]

### 2.2 FILTER WITH GIANT COMPONENT
tmp_net = graph_from_data_frame(d=filter_net_edge, vertices=filter_net_node, directed=F)
all_components = groups(components(tmp_net))
# COLLECT ALL LARGE COMPONENTS
giant_comp_node = c()
giant_comp_threshold = 20
for (x in 1:length(all_components)){
  each_comp = all_components[[x]]
  if (length(each_comp) >= giant_comp_threshold){
    giant_comp_node = c(giant_comp_node, each_comp)
  }
}
  
refilter_net_edge<-subset(filter_net_edge, (From %in% giant_comp_node | To %in% giant_comp_node))
refilter_net_edge_node = unique(c(refilter_net_edge$From, refilter_net_edge$To))
refilter_net_node = filter_net_node[filter_net_node$Gene_num %in% refilter_net_edge_node,]

### 3. BUILD UP GRAPH
net = graph_from_data_frame(d=refilter_net_edge, vertices=refilter_net_node, directed=F)

# refilter_net_node$att_deg = strength(net, weights=E(net)$Attention)
### 4. NETWORK PARAMETERS SETTINGS
# vertex frame color
vertex_fcol = rep('black', vcount(net))
# vertex color
vertex_col = rep('mediumpurple1', vcount(net))
vertex_col[V(net)$NodeType=='Gene-METH'] = 'plum1'
vertex_col[V(net)$NodeType=='Gene-PROT'] = 'gray'
# vertex size
vertex_size = rep(5.0, vcount(net))
# vertex cex
vertex_cex = rep(0.5, vcount(net))
# edge width
edge_width = rep(2, ecount(net))
# edge color
edge_color = rep('gray', ecount(net))
edge_color[E(net)$EdgeType=='Gene-TRAN-Gene-PROT'] = 'mediumpurple1'
edge_color[E(net)$EdgeType=='Gene-TRAN-Gene-METH'] = 'plum1'


set.seed(18)
plot(net,
     vertex.frame.width = 0.1,
     vertex.frame.color = vertex_fcol,
     vertex.color = vertex_col,
     vertex.size = vertex_size,
    #  vertex.shape = c('square', 'circle')[1+(V(net)$NodeType=='gene')],
     vertex.label = V(net)$Gene_name,
     vertex.label.color = 'black',
     vertex.label.cex = vertex_cex,
     edge.width = edge_width,
     edge.color = edge_color,
     edge.curved = 0.2,
     layout=layout_nicely)
### ADD LEGEND
legend(x=-1.15, y= -0.3, legend=c('Genes', 'Promoters', 'Proteins'), pch=c(21,21,21),
       pt.bg=c('mediumpurple1', 'plum1', 'gray'), pt.cex=4.5, cex=0.8, bty='n')
legend(x=-1.15, y= -0.7,
       legend=c('Protein-Protein', 'Gene-Protein', 'Promoter-Gene'),
       col=c('gray', 'mediumpurple1', 'plum1'), lwd=c(4.5, 4.5), cex=0.8, bty='n')


# layout=layout_nicely
# layout=layout_as_tree
# layout_with_kk
# layout=layout_with_dh
# layout=layout_with_gem



