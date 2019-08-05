library(Seurat)
library(dyno)
library(ggplot2)
plot_data <- read.csv('/home/ivan/Desktop/Project2/MyData/paper/full_mature_exdata.csv',header=TRUE,sep=",")
rownames(plot_data) = plot_data$cell
colnames(plot_data)
factors <- read.csv('/home/ivan/Desktop/Project2/MyData/paper/factors.csv',header=TRUE,sep=",")
new_data = t(plot_data[,2:9961])
rownames(factors) = factors$cell
seurat_paper <- CreateSeuratObject(counts = new_data,meta.data=factors, min.cells = 3, min.features = 1, project = "my_scRNAseq")
Idents(seurat_paper) <- "group"
dim(seurat_paper)
lst_t=c()
group_1=c('CD69-DP','CD69+DP','CD4+CD8low','CD4SP_mature','CD8SP')
for (i in 1:(length(group_1)-1)){
  for (j in (i+1):length(group_1)){
    markers.all = FindMarkers(seurat_paper,assay="RNA",ident.1=group_1[i],ident.2=group_1[j])
    lst1=rownames(markers.all[order(markers.all['p_val'])[1:24],])
    lst_t = unique(c(lst1,lst_t))
  }
}
length(lst_t)

write.table(lst_t, '/home/ivan/Desktop/Project2/MyData/pipeline/seurat_degene.txt', row.names=FALSE, quote=FALSE, sep='\t')

lst_t[!lst_t %in% colnames(t(new_data))]

lst_drode <- read.csv('/home/ivan/Desktop/Project2/MyData/pipeline/degenes_drode.csv',header=TRUE,sep=",")
lst_drode = c(as.vector(lst_drode[,2]))

for(i in 1:length(lst_drode)){
  if(!lst_drode[i] %in% colnames(t(new_data))){
    print(lst_drode[i])
    lst_drode[i]=gsub("-", ".", lst_drode[i])
  }
  if(!lst_drode[i] %in% colnames(t(new_data))){
    print('wrong')
  }
}

dataset <- wrap_expression(
  counts = t(new_data)[,lst_drode],
  expression = t(new_data)[,lst_drode]
)

model <- infer_trajectory(dataset, 'slingshot')
model <- model %>% add_dimred(dyndimred::dimred_mds, expression_source = dataset$expression)
plot_dimred(
  model, 
  expression_source = dataset$expression, 
  grouping = factors$group
)
