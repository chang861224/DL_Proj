def output(y_pred, output_path):
    output="article_id\tstart_position\tend_position\tentity_text\tentity_type\n"
    for test_id in range(len(y_pred)):
        pos=0
        start_pos=None
        end_pos=None
        entity_text=None
        entity_type=None
        for pred_id in range(len(y_pred[test_id])):
            if y_pred[test_id][pred_id][0]=='B':
                start_pos=pos
                entity_type=y_pred[test_id][pred_id][2:]
            elif start_pos is not None and y_pred[test_id][pred_id][0]=='I' and y_pred[test_id][pred_id+1][0]=='O':
                end_pos=pos
                entity_text=''.join([testdata_list[test_id][position][0] for position in range(start_pos,end_pos+1)])
                line=str(testdata_article_id_list[test_id])+'\t'+str(start_pos)+'\t'+str(end_pos+1)+'\t'+entity_text+'\t'+entity_type
                output+=line+'\n'
            pos+=1

    #output_path='output.tsv'
    with open(output_path,'w',encoding='utf-8') as f:
        f.write(output)
