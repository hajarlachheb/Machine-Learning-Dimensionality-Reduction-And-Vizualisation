import numpy as np
import random
import load_datasets

X, y = load_datasets.load_vowel()
df_array = np.array(X)


def init_centroids(k, ar):

    dim = len(ar[0])

    centroids = []
    rand_idxs = random.sample(range(0, len(ar)), k)
    centroids = [ar[rand_i] for rand_i in rand_idxs]
    return centroids


def euclidean_distance(a, b):

    dist = np.linalg.norm(a - b)
    return dist


def assign_cluster(seeds, ar):

    centroids = []


    for idx,x in enumerate(ar):
        error = []
        #calculate distance of each datapoint to the centroids
        for jdx,seed in enumerate(seeds):
            error.append(euclidean_distance(x,seed))
        # take the minimum distance and assign it as the cluster
        centroids.append(np.argmin(error))

    return np.array(centroids)


def update_centroids(k, centroids, assigned_to, X, algorithm='means'):
    mean_centroids = []
    output_centroids = []
    for cidx in range(k):
    #for assigned in list(set(assigned_to)):

        mean_centroid = []
        counter = 0
        for i,x in enumerate(X):
            # find the current cluster
            if assigned_to[i] == cidx:
                mean_centroid.append(x)
                counter += 1
        # if centroid has no membership add the old centroid.
        if not mean_centroid:
            mean_centroids.append(centroids[cidx])
        else:
            if algorithm == 'means':
                mean_centroids.append(np.array(mean_centroid).mean(axis=0))
                #mean_centroids.append(np.sum(mean_centroid)/counter)


            if algorithm == 'harmonic': #k-harmonic-means
                #avoid division by zero
                epsilon = 0.0001
                mean_centroids.append(counter/sum(1/(np.array(mean_centroid)+epsilon)))

    return mean_centroids


def kmeans(k, ar, algorithm):

    centroids = init_centroids(k, ar)

    assigned_points = assign_cluster(centroids, ar)
    iterations = 0
    saved_centroids = []
    while(iterations < 50):
        previous_centroids = centroids
        prev_points = assigned_points
        assigned_points = assign_cluster(centroids, ar)
        centroids = update_centroids(k,previous_centroids,assigned_points,ar,algorithm)

        # compute error
        errors_centroids = []
        for i in range(len(centroids)):
            for j in range(i + 1, len(centroids)):
                errors_centroids.append(euclidean_distance(centroids[i], centroids[j]))



        deviations = [float(euclidean_distance(previous_centroids[i],centroids[i])) for i in range(len(previous_centroids))]

        # break condition if deviation to previous clusters is small

        iterations += 1
#        print('devs:' ,deviations)
#        print(sum(deviations))
        #print('itereration',iterations, 'errors: ',errors_centroids)
        if sum(deviations) < 1e-7:
#            print('stop')
            break
    assigned_points = assign_cluster(centroids, ar)
    #print("num_iter: ",iterations)
    return centroids,assigned_points


def performance_score(X, centroids):
    #inter cluster distance
    inter_dist = []
    for i,c in enumerate(centroids):
        for j in range(i+1,len(centroids)):
            inter_dist.append(euclidean_distance(c,centroids[j]))

    mean_inter = np.array(inter_dist).mean(axis=0)

    # intra cluster distance

    intra_dists = []

    for jdx, centroid in enumerate(centroids):

        dist = []
        # calculate distance of each datapoint to the centroids
        for idx, x in enumerate(X):
            dist.append(euclidean_distance(x, centroid))

        intra_dists.append(np.mean(dist))


    mean_intra = np.mean(intra_dists)

    # inter cluster distance should be big intra cluster distance should be small
    # given the final calculation a high score indicates a good score and a low score a bad score
    return round(mean_inter-mean_intra,4)

