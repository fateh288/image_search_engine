#     # task_write_set1()
#     # task_write_set2()
#     # task_write_set3()
# def olivetti():
#     dataset_images = data_loader_utils.load_olivetti_dataset_images()
#     #print(dataset['DESCR'])
#     image1 = dataset_images[0]
#     color_moment1 = ColorMoment(image1, 8)
#     print(color_moment1)
#     image2 = dataset_images[1]
#     color_moment2 = ColorMoment(image2, 8)
#     image3 = dataset_images[2]
#     color_moment3 = ColorMoment(image3, 8)
#
#     sim12 = similarity_measures.color_moment_similarity(color_moment1,color_moment2)
#     sim13 = similarity_measures.color_moment_similarity(color_moment1,color_moment3)
#     print("sim12=",sim12)
#     print("sim13=",sim13)
#     print("sim23=",similarity_measures.color_moment_similarity(color_moment2,color_moment3))
#
#     images = [image2,image3,image2,image3, image3,image3, image3,image3, image3, image3 ]
#     query_image_data = 0
#     similar_images_data = [[1,sim12],[2,sim13],[1,sim12],[2,sim13], [2,sim13], [2,sim13], [2,sim13], [2,sim13], [2,sim13], [2,sim13]]
#     display_utils.display_similar_images(image1,images,query_image_data,similar_images_data)
#
#     elbp1 = ExtendedLocalBinaryPattern(image1)
#     print(image1, flush=True)
#     print(image1.shape)
#     print(type(image2))
#     lbp_feature = elbp1.lbp()
#     print("lbp_shape=",lbp_feature.shape)
#     #print(image1[0:5,0:5],"\n",lbp_feature[0:5,0:5])
#     print("max lbp=",np.max(lbp_feature)," min_lbp=",np.min(lbp_feature),"median_lbp=",np.median(lbp_feature))
#
#     hog1 = HOG(image1)
#     hog_feature1 = hog1.hog_feature()
#     #print(hog_feature1)
#
#     hog2 = HOG(image2)
#     hog_feature2 = hog2.hog_feature()
#
#     image12 = dataset_images[11]
#     hog12 = HOG(image12)
#     hog_feature12 = hog12.hog_feature()
#
#     #l1_1_2 = similarity_measures.l_norm_similarity(hog_feature1,hog_feature2, 1)
#     #l2_1_2 = similarity_measures.l_norm_similarity(hog_feature1,hog_feature2, 2)
#     #l5_1_2 = similarity_measures.l_norm_similarity(hog_feature1, hog_feature2, 5)
#
#     #l2_1_12 = similarity_measures.l_norm_similarity(hog_feature1, hog_feature12, 2)
#
#     #display_utils.display_similar_images(image1, [image2,image12], 0, [[1,l2_1_2],[10,l2_1_12]])
#
#     feature_list1 = FeatureList(image1,1)
#     feature_list2 = FeatureList(image2,2)
#     feature_list3 = FeatureList(image3,3)
#     feature_list1.add_default_features()
#     feature_list2.add_default_features()
#     feature_list3.add_default_features()
#
#
#     shelve_db = database_utils.ShelveDB("olivetti")
#     shelve_db.add_object("1", feature_list1)
#     shelve_db.add_object("2", feature_list2)
#     shelve_db.add_object("3", feature_list3)
#
#     shelve_db.print_db()
#
# def cat_write():
#     dataset = data_loader_utils.load_dataset_from_folder("cats",'.jpg')
#     tasks.store_image_features_in_database('cats',dataset['images'],dataset['ids'])
#     tasks.write_database_to_file('cats','cats_db.txt')
#     #print(dataset['images'].shape)
#     #print(dataset['ids'].shape)
#
# def cat_read():
#     tasks.print_feature(dataset_name='cats',id=10,feature_type="color")
#     tasks.print_feature(dataset_name='cats', id=10, feature_type="elbp")
#     tasks.print_feature(dataset_name='cats', id=10, feature_type="hog")
#
# def cat_similarity():
#     tasks.get_closest('cats',10,'hog')
#
# def task_write_set1():
#     dataset = data_loader_utils.load_dataset_from_folder("set1",'.png')
#     tasks.store_image_features_in_database('set1',dataset['images'],dataset['ids'])
#     tasks.write_database_to_file('set1','set1_db.txt')
#
# def task_write_set2():
#     dataset = data_loader_utils.load_dataset_from_folder("set2",'.png')
#     tasks.store_image_features_in_database('set2',dataset['images'],dataset['ids'])
#     tasks.write_database_to_file('set2','set2_db.txt')
#
# def task_write_set3():
#     dataset = data_loader_utils.load_dataset_from_folder("set3",'.png')
#     tasks.store_image_features_in_database('set3',dataset['images'],dataset['ids'])
#     tasks.write_database_to_file('set3','set3_db.txt')
# task_similarity('set2', 'image-10', 'hog')
# task_similarity('set2','image-10','elbp')
# task_similarity('set2', 'image-10', 'all')
# task_similarity('set3', 'image-70', 'color')
# task_similarity('set3', 'image-70', 'hog')
# task_similarity('set3', 'image-70', 'elbp')
# task_similarity('set3', 'image-70', 'all')
# shelve_db = database_utils.ShelveDB('set1')
# query_feature = shelve_db.select(['image-8'])[0]
# to_draw_image = query_feature.image
# display_utils.display_images([to_draw_image])
# for block_id in range(0,64):
#     start, end = utils.get_coord_from_block_id(block_id, 8, 8)
#     to_draw_image = display_utils.draw(to_draw_image, start_coord=start, end_coord=end, val=0.1)
#     #query_image = display_utils.draw(query_image, start_coord=start, end_coord=end, val=colors[index])
# display_utils.display_images([to_draw_image])
# for i in range(0,64):
#     utils.get_coord_from_block_id(i,8,8)
# cat_write()
# cat_read()
# cat_similarity()
# olivetti()
# shelve = database_utils.ShelveDB("olivetti")
# shelve.print_db()
# res = shelve.select()
# print(res)
# print(res)