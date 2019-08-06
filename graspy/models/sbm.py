import numpy as np
from sklearn.utils import check_X_y
from sklearn.cluster import AgglomerativeClustering

from ..cluster import GaussianCluster
from ..embed import AdjacencySpectralEmbed, LaplacianSpectralEmbed
from ..utils import (
    augment_diagonal,
    cartprod,
    import_graph,
    is_unweighted,
    remove_loops,
    symmetrize,
)
from .base import BaseGraphEstimator, _calculate_p


def _check_common_inputs(n_components, min_comm, max_comm, cluster_kws, embed_kws):
    if not isinstance(n_components, int) and n_components is not None:
        raise TypeError("n_components must be an int or None")
    elif n_components is not None and n_components < 1:
        raise ValueError("n_components must be > 0")

    if not isinstance(min_comm, int):
        raise TypeError("min_comm must be an int")
    elif min_comm < 1:
        raise ValueError("min_comm must be > 0")

    if not isinstance(max_comm, int):
        raise TypeError("max_comm must be an int")
    elif max_comm < 1:
        raise ValueError("max_comm must be > 0")
    elif max_comm < min_comm:
        raise ValueError("max_comm must be >= min_comm")

    if not isinstance(cluster_kws, dict):
        raise TypeError("cluster_kws must be a dict")

    if not isinstance(embed_kws, dict):
        raise TypeError("embed_kws must be a dict")


class SBMEstimator(BaseGraphEstimator):
    r"""
    Stochastic Block Model 

    The stochastic block model (SBM) represents each node as belonging to a block 
    (or community). For a given potential edge between node :math:`i` and :math:`j`, 
    the probability of an edge existing is specified by the block that nodes :math:`i`
    and :math:`j` belong to:

    :math:`P_{ij} = B_{\tau_i \tau_j}`

    where :math:`B \in \mathbb{[0, 1]}^{K x K}` and :math:`\tau` is an `n\_nodes` 
    length vector specifying which block each node belongs to. 

    Read more in the :ref:`tutorials <models_tutorials>`

    Parameters
    ----------
    directed : boolean, optional (default=True)
        Whether to treat the input graph as directed. Even if a directed graph is inupt, 
        this determines whether to force symmetry upon the block probability matrix fit
        for the SBM. It will also determine whether graphs sampled from the model are 
        directed. 

    loops : boolean, optional (default=False)
        Whether to allow entries on the diagonal of the adjacency matrix, i.e. loops in 
        the graph where a node connects to itself. 

    n_components : int, optional (default=None)
        Desired dimensionality of embedding for clustering to find communities.
        ``n_components`` must be ``< min(X.shape)``. If None, then optimal dimensions 
        will be chosen by :func:`~graspy.embed.select_dimension``.

    min_comm : int, optional (default=1)
        The minimum number of communities (blocks) to consider. 

    max_comm : int, optional (default=10)
        The maximum number of communities (blocks) to consider (inclusive).

    cluster_kws : dict, optional (default={})
        Additional kwargs passed down to :class:`~graspy.cluster.GaussianCluster`
    
    embed_kws : dict, optional (default={})
        Additional kwargs passed down to :class:`~graspy.embed.AdjacencySpectralEmbed`

    Attributes
    ----------
    block_p_ : np.ndarray, shape (n_blocks, n_blocks)
        The block probability matrix :math:`B`, where the element :math:`B_{i, j}`
        represents the probability of an edge between block :math:`i` and block 
        :math:`j`.

    p_mat_ : np.ndarray, shape (n_verts, n_verts)
        Probability matrix :math:`P` for the fit model, from which graphs could be
        sampled.

    vertex_assignments_ : np.ndarray, shape (n_verts)
        A vector of integer labels corresponding to the predicted block that each node 
        belongs to if ``y`` was not passed during the call to ``fit``. 

    block_weights_ : np.ndarray, shape (n_blocks)
        Contains the proportion of nodes that belong to each block in the fit model.

    See also
    --------
    graspy.models.DCSBMEstimator
    graspy.simulations.sbm

    References
    ----------
    .. [1]  Holland, P. W., Laskey, K. B., & Leinhardt, S. (1983). Stochastic
            blockmodels: First steps. Social networks, 5(2), 109-137.
    """

    def __init__(
        self,
        directed=True,
        loops=False,
        n_components=None,
        min_comm=1,
        max_comm=10,
        cluster_kws={},
        embed_kws={},
    ):
        super().__init__(directed=directed, loops=loops)

        _check_common_inputs(n_components, min_comm, max_comm, cluster_kws, embed_kws)

        self.cluster_kws = cluster_kws
        self.n_components = n_components
        self.min_comm = min_comm
        self.max_comm = max_comm
        self.embed_kws = embed_kws

    def _estimate_assignments(self, graph):
        """
        Do some kind of clustering algorithm to estimate communities

        There are many ways to do this, here is one
        """
        embed_graph = augment_diagonal(graph)
        latent = AdjacencySpectralEmbed(
            n_components=self.n_components, **self.embed_kws
        ).fit_transform(embed_graph)
        if isinstance(latent, tuple):
            latent = np.concatenate(latent, axis=1)
        gc = GaussianCluster(
            min_components=self.min_comm,
            max_components=self.max_comm,
            **self.cluster_kws
        )
        vertex_assignments = gc.fit_predict(latent)
        self.vertex_assignments_ = vertex_assignments

    def fit(self, graph, y=None):
        """
        Fit the SBM to a graph, optionally with known block labels

        If y is `None`, the block assignments for each vertex will first be
        estimated.

        Parameters
        ----------
        graph : array_like or networkx.Graph
            Input graph to fit

        y : array_like, length graph.shape[0], optional
            Categorical labels for the block assignments of the graph

        """
        graph = import_graph(graph)

        if not is_unweighted(graph):
            raise NotImplementedError(
                "Graph model is currently only implemented for unweighted graphs."
            )

        if y is None:
            self._estimate_assignments(graph)
            y = self.vertex_assignments_

            _, counts = np.unique(y, return_counts=True)
            self.block_weights_ = counts / graph.shape[0]
        else:
            check_X_y(graph, y)

        block_vert_inds, block_inds, block_inv = _get_block_indices(y)

        if not self.loops:
            graph = remove_loops(graph)
        block_p = _calculate_block_p(graph, block_inds, block_vert_inds)

        if not self.directed:
            block_p = symmetrize(block_p)
        self.block_p_ = block_p

        p_mat = _block_to_full(block_p, block_inv, graph.shape)
        if not self.loops:
            p_mat = remove_loops(p_mat)
        self.p_mat_ = p_mat

        return self

    def _n_parameters(self):
        n_blocks = self.block_p_.shape[0]
        n_parameters = 0
        if self.directed:
            n_parameters += n_blocks ** 2
        else:
            n_parameters += n_blocks * (n_blocks + 1) / 2
        if hasattr(self, "vertex_assignments_"):
            n_parameters += n_blocks - 1
        return n_parameters


class DCSBMEstimator(BaseGraphEstimator):
    r"""
    Degree-corrected Stochastic Block Model

    The degree-corrected stochastic block model (DCSBM) represents each node as 
    belonging to a block (or community). For a given potential edge between node
    :math:`i` and :math:`j`, the probability of an edge existing is specified by 
    the block that nodes :math:`i` and :math:`j` belong to as in the SBM. However,
    an additional "promiscuity" parameter :math:`\theta` is added for each node, 
    allowing the vertices within a block to have heterogeneous expected degree 
    distributions: 

    :math:`P_{ij} = \theta_i \theta_j B_{\tau_i \tau_j}`

    where :math:`B \in \mathbb{[0, 1]}^{K x K}` :math:`\tau` is an `n\_nodes` 
    length vector specifying which block each node belongs to, and :math:`\theta`
    is an `n\_nodes` length vector specifiying the degree correction for each
    node. 

    The ``degree_directed`` parameter of this model allows the degree correction 
    parameter to be different for the in and out degree of each node:  

    :math:`P_{ij} = \theta_i \eta_j B_{\tau_i \tau_j}`

    where :math:`\theta` and :math:`\eta` need not be the same.
    
    Read more in the :ref:`tutorials <models_tutorials>`

    Parameters
    ----------
    directed : boolean, optional (default=True)
        Whether to treat the input graph as directed. Even if a directed graph is inupt, 
        this determines whether to force symmetry upon the block probability matrix fit
        for the SBM. It will also determine whether graphs sampled from the model are 
        directed. 

    degree_directed : boolean, optional (default=False)
        Whether to fit an "in" and "out" degree correction for each node. In the
        degree_directed case, the fit model can have a different expected in and out 
        degree for each node. 

    loops : boolean, optional (default=False)
        Whether to allow entries on the diagonal of the adjacency matrix, i.e. loops in 
        the graph where a node connects to itself. 

    n_components : int, optional (default=None)
        Desired dimensionality of embedding for clustering to find communities.
        ``n_components`` must be ``< min(X.shape)``. If None, then optimal dimensions 
        will be chosen by :func:`~graspy.embed.select_dimension``.

    min_comm : int, optional (default=1)
        The minimum number of communities (blocks) to consider. 

    max_comm : int, optional (default=10)
        The maximum number of communities (blocks) to consider (inclusive).

    cluster_kws : dict, optional (default={})
        Additional kwargs passed down to :class:`~graspy.cluster.GaussianCluster`
    
    embed_kws : dict, optional (default={})
        Additional kwargs passed down to :class:`~graspy.embed.LaplacianSpectralEmbed`

    Attributes
    ----------
    block_p_ : np.ndarray, shape (n_blocks, n_blocks)
        The block probability matrix :math:`B`, where the element :math:`B_{i, j}`
        represents the expected number of edges between block :math:`i` and block 
        :math:`j`.

    p_mat_ : np.ndarray, shape (n_verts, n_verts)
        Probability matrix :math:`P` for the fit model, from which graphs could be
        sampled.

    degree_corrections_ : np.ndarray, shape (n_verts, 1) or (n_verts, 2)
        Degree correction vector(s) :math:`theta`. If `degree_directed` parameter was
        False, then will be of shape (n_verts, 1) and element :math:`i` represents the 
        degree correction for node :math:`i`. Otherwise, the first column contains out 
        degree corrections and the second column contains in degree corrections. 

    vertex_assignments_ : np.ndarray, shape (n_verts)
        A vector of integer labels corresponding to the predicted block that each node 
        belongs to if ``y`` was not passed during the call to ``fit``. 

    block_weights_ : np.ndarray, shape (n_blocks)
        Contains the proportion of nodes that belong to each block in the fit model.

    See also
    --------
    graspy.models.SBMEstimator
    graspy.simulations.sbm

    Notes
    -----
    Note that many examples in the literature describe the DCSBM as being sampled with a 
    Poisson distribution. Here, we implement this model with a Bernoulli. When 
    individual edge probabilities are relatively low these two distributions will yield 
    similar results. 

    References
    ----------
    .. [1]  Karrer, B., & Newman, M. E. (2011). Stochastic blockmodels and community
            structure in networks. Physical review E, 83(1), 016107.
    """

    def __init__(
        self,
        degree_directed=False,
        directed=True,
        loops=False,
        n_components=None,
        min_comm=1,
        max_comm=10,
        cluster_kws={},
        embed_kws={},
    ):
        super().__init__(directed=directed, loops=loops)
        _check_common_inputs(n_components, min_comm, max_comm, cluster_kws, embed_kws)

        if not isinstance(degree_directed, bool):
            raise TypeError("`degree_directed` must be of type bool")

        self.degree_directed = degree_directed
        self.cluster_kws = cluster_kws
        self.n_components = n_components
        self.min_comm = min_comm
        self.max_comm = max_comm
        self.embed_kws = embed_kws

    def _estimate_assignments(self, graph):
        graph = symmetrize(graph, method="avg")  # TODO use directed LSE
        lse = LaplacianSpectralEmbed(
            form="R-DAD", n_components=self.n_components, **self.embed_kws
        )
        latent = lse.fit_transform(graph)
        gc = GaussianCluster(
            min_components=self.min_comm,
            max_components=self.max_comm,
            **self.cluster_kws
        )
        self.vertex_assignments_ = gc.fit_predict(latent)

    def fit(self, graph, y=None):
        """
        Fit the DCSBM to a graph, optionally with known block labels

        If y is `None`, the block assignments for each vertex will first be
        estimated.

        Parameters
        ----------
        graph : array_like or networkx.Graph
            Input graph to fit

        y : array_like, length graph.shape[0], optional
            Categorical labels for the block assignments of the graph

        Returns
        -------
        self : ``DCSBMEstimator`` object 
            Fitted instance of self 
        """
        graph = import_graph(graph)
        if y is None:
            self._estimate_assignments(graph)
            y = self.vertex_assignments_
            _, counts = np.unique(y, return_counts=True)
            self.block_weights_ = counts / graph.shape[0]

        block_vert_inds, block_inds, block_inv = _get_block_indices(y)

        if not self.loops:
            graph = graph - np.diag(np.diag(graph))
        block_p = _calculate_block_p(graph, block_inds, block_vert_inds)

        out_degree = np.count_nonzero(graph, axis=1).astype(float)
        in_degree = np.count_nonzero(graph, axis=0).astype(float)
        if self.degree_directed:
            degree_corrections = np.stack((out_degree, in_degree), axis=1)
        else:
            degree_corrections = (out_degree + in_degree) / 2
            # new axis just so we can index later
            degree_corrections = degree_corrections[:, np.newaxis]
        for i in block_inds:
            block_degrees = degree_corrections[block_vert_inds[i]]
            degree_divisor = np.sum(block_degrees, axis=0)
            if not isinstance(degree_divisor, np.float64):
                degree_divisor[degree_divisor == 0] = 1
            degree_corrections[block_vert_inds[i]] = (
                degree_corrections[block_vert_inds[i]] / degree_divisor
            )
        self.degree_corrections_ = degree_corrections

        block_p = _calculate_block_p(
            graph, block_inds, block_vert_inds, return_counts=True
        )
        p_mat = _block_to_full(block_p, block_inv, graph.shape)
        p_mat = p_mat * np.outer(degree_corrections[:, 0], degree_corrections[:, -1])

        if not self.loops:
            p_mat -= np.diag(np.diag(p_mat))
        self.p_mat_ = p_mat
        self.block_p_ = block_p
        return self

    def _n_parameters(self):
        n_blocks = self.block_p_.shape[0]
        n_parameters = 0
        if self.directed:
            n_parameters += n_blocks ** 2  # B matrix
        else:
            n_parameters += n_blocks * (n_blocks + 1) / 2  # Undirected B matrix
        if hasattr(self, "vertex_assignments_"):
            n_parameters += n_blocks - 1
        n_parameters += self.degree_corrections_.size
        return n_parameters


def _get_block_indices(y):
    """
    y is a length n_verts vector of labels

    returns a length n_verts vector in the same order as the input
    indicates which block each node is
    """
    block_labels, block_inv, block_sizes = np.unique(
        y, return_inverse=True, return_counts=True
    )

    n_blocks = len(block_labels)
    block_inds = range(n_blocks)

    block_vert_inds = []
    for i in block_inds:
        # get the inds from the original graph
        inds = np.where(block_inv == i)[0]
        block_vert_inds.append(inds)
    return block_vert_inds, block_inds, block_inv


def _calculate_block_p(graph, block_inds, block_vert_inds, return_counts=False):
    """
    graph : input n x n graph 
    block_inds : list of length n_communities
    block_vert_inds : list of list, for each block index, gives every node in that block
    return_counts : whether to calculate counts rather than proportions
    """

    n_blocks = len(block_inds)
    block_pairs = cartprod(block_inds, block_inds)
    block_p = np.zeros((n_blocks, n_blocks))

    for p in block_pairs:
        from_block = p[0]
        to_block = p[1]
        from_inds = block_vert_inds[from_block]
        to_inds = block_vert_inds[to_block]
        block = graph[from_inds, :][:, to_inds]
        if return_counts:
            p = np.count_nonzero(block)
        else:
            p = _calculate_p(block)
        block_p[from_block, to_block] = p
    return block_p


def _block_to_full(block_mat, inverse, shape):
    """
    "blows up" a k x k matrix, where k is the number of communities, 
    into a full n x n probability matrix

    block mat : k x k 
    inverse : array like length n, 
    """
    block_map = cartprod(inverse, inverse).T
    mat_by_edge = block_mat[block_map[0], block_map[1]]
    full_mat = mat_by_edge.reshape(shape)
    return full_mat


class HSBMEstimator(SBMEstimator):
    def __init__(
        self,
        n_levels=2,
        cluster_method="gmm",
        embed_method="ase",
        cluster_kws={},
        embed_kws={},
        diag_aug_weight=1,
        n_components=None,
        n_subgraphs=2,
        bandwidth=None,
    ):
        self.n_levels = n_levels
        self.cluster_method = cluster_method
        self.embed_method = embed_method
        self.n_components = n_components
        self.n_subgraphs = n_subgraphs
        self.bandwidth = bandwidth
        self.diag_aug_weight = diag_aug_weight
        self.embed_kws = embed_kws
        self.cluster_kws = cluster_kws

    def fit(self, graph, y=None):
        embed_graph = augment_diagonal(graph, weight=self.diag_aug_weight)
        if self.embed_method == "ase":
            embed = AdjacencySpectralEmbed(
                n_components=self.n_components, **self.embed_kws
            )
        elif self.embed_method == "lse":
            embed = LaplacianSpectralEmbed(
                n_components=self.n_components, **self.embed_kws
            )

        latent = embed.fit_transform(embed_graph)
        if isinstance(latent, tuple):
            latent = np.concatenate(latent, axis=-1)

        # TODO : Probably make this normalization an option. Need to figure out when
        #        this matters (e.g. even a non-dc hsbm that does have differing degrees
        #        just due to block structure gets messed up by this)
        latent = latent / np.linalg.norm(latent, axis=1)[:, np.newaxis]

        # TODO : If doing the degree normalization, should we do something smarter to
        #        consider geodesic distances on the unit ball?

        if self.cluster_method == "gmm":
            cluster = GaussianCluster(
                min_components=1, max_components=self.n_subgraphs, **self.cluster_kws
            )
        # TODO : could also do kmeans here

        # TODO : this clustering should probably use many random inits and find the best
        #        on some metric
        vertex_assignments = cluster.fit(latent).model_.predict(latent)

        sub_vert_inds, sub_inds, sub_inv = _get_block_indices(vertex_assignments)

        subgraph_latents = []
        for inds in sub_vert_inds:
            subgraph = graph[np.ix_(inds, inds)]
            # TODO : how to choose the number of components to embed each subgraph into?
            #        I think they need to see the same, but check this. One option is ZG
            #        and then take the max for all subgraphs
            embed = AdjacencySpectralEmbed(n_components=3, **self.embed_kws)
            sublatent = embed.fit_transform(subgraph)
            if isinstance(sublatent, tuple):
                sublatent = np.concatenate(sublatent, axis=-1)
            subgraph_latents.append(sublatent)

        # TODO : do we know this kernel works still, considering the W rotation matrix?
        # TODO : consider a MGC or DCorr kernel here?
        subgraph_dissimilarities = _compute_subgraph_dissimilarities(
            subgraph_latents, sub_inds, self.bandwidth
        )

        # TODO : how to choose the number of clusters here for the agglomeration step?
        #        This will yield the number of motifs in the graph, at one level below
        # TODO : implement grid sweep agglomerative. sweep the linkages, in this case
        #        no randomness is needed here, I think. Could also compare multiple
        #        embedding dims tho
        agglom = AgglomerativeClustering(
            n_clusters=3, affinity="precomputed", linkage="average"
        )
        subgraph_short_labels = agglom.fit_predict(subgraph_dissimilarities)
        subgraph_labels = subgraph_short_labels[sub_inv]

        # TODO : given the labels, can compute actual probability matrices and B mats
        # TODO : recursive step
        return vertex_assignments, subgraph_labels


def _compute_subgraph_dissimilarities(subgraph_latents, subgraph_inds, bandwidth):
    n_subgraphs = len(subgraph_inds)
    subgraph_pairs = cartprod(subgraph_inds, subgraph_inds)
    subgraph_dissimilarities = np.zeros((n_subgraphs, n_subgraphs))

    for p in subgraph_pairs:
        sub1 = p[0]
        sub2 = p[1]
        latent1 = subgraph_latents[sub1]
        latent2 = subgraph_latents[sub2]
        t_stat = _mmd_kernel(latent1, latent2, bandwidth)
        subgraph_dissimilarities[sub1, sub2] = t_stat

    return subgraph_dissimilarities


def _gaussian_covariance(X, Y, bandwidth):
    diffs = np.expand_dims(X, 1) - np.expand_dims(Y, 0)
    if bandwidth is None:
        bandwidth = 0.5
    return np.exp(-0.5 * np.sum(diffs ** 2, axis=2) / bandwidth ** 2)


def _mmd_kernel(X, Y, bandwidth):
    N, _ = X.shape
    M, _ = Y.shape
    x_stat = np.sum(_gaussian_covariance(X, X, bandwidth) - np.eye(N)) / (N * (N - 1))
    y_stat = np.sum(_gaussian_covariance(Y, Y, bandwidth) - np.eye(M)) / (M * (M - 1))
    xy_stat = np.sum(_gaussian_covariance(X, Y, bandwidth)) / (N * M)
    return x_stat - 2 * xy_stat + y_stat
