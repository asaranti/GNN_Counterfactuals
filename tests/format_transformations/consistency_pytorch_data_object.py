"""
    Check the consistency of any transformation that return a
    pytorch_geometric.data.Data object

    :author: Anna Saranti
    :copyright: © 2021 HCI-KDD (ex-AI) group
    :date: 2021-12-15
"""

import os
import time
import unittest

from HTMLTestRunner import HTMLTestRunner

from preprocessing.format_transformation_ppi_to_pytorch import transform_from_ppi_to_pytorch
from preprocessing.format_transformation_pytorch_to_ui import transform_from_pytorch_to_ui
from preprocessing.format_transformation_ui_to_pytorch import transform_from_ui_to_pytorch


class DataConsistency(unittest.TestCase):
    """
    Check the consistency of any transformation that return a
    pytorch_geometric.data.Data object. The number of nodes, edges,
    their attributes and ids must be consistent. For example,
    one cannot have edge_index=[2, 10] and edge_attr=[25, 5], because
    the number of edges (10 and 25) do not match.
    """

    def setUp(self):
        """
        Set Up the text fixture - the common initialization
        of objects used in the unit tests.
        """

        # [1.] Apply a transformation from the ppi format to a pytorch graph format ------------------------------------
        dataset_folder = os.path.join("data", "Protein_Dataset")
        pytorch_ppi_attributes_file = \
            "Human__TCGA_ACC__UNC__RNAseq__HiSeq_RNA__01_28_2016__BI__Gene__Firehose_RSEM_log2.cct"
        pytorch_ppi_node_id_to_name_file = "human.name_2_string.csv"
        pytorch_ppi_edges_file = "9606.protein.links.v11.0.txt"

        self.ppi_pytorch_graph = transform_from_ppi_to_pytorch(dataset_folder,
                                                               pytorch_ppi_attributes_file,
                                                               pytorch_ppi_node_id_to_name_file,
                                                               pytorch_ppi_edges_file)

    def test_nodes_nr(self):
        """
        Test the nodes number consistency over all internal data structures of the pytorch Data object
        that involve nodes. The corresponding axes must have an equal number of nodes >= 1
        """

        self.assertEqual(self.ppi_pytorch_graph.x.size(dim=0), self.ppi_pytorch_graph.node_labels.shape[0],
                         "graph.x.size(dim=0) and graph.node_labels.shape[0] must be equal")
        self.assertEqual(self.ppi_pytorch_graph.x.size(dim=0), self.ppi_pytorch_graph.node_ids.shape[0],
                         "graph.x.size(dim=0) and graph.node_ids.shape[0] must be equal")

        self.assertGreater(self.ppi_pytorch_graph.x.size(dim=0), 1, "The number of nodes must be greater than one.")

    def test_node_attr_nr(self):
        """
        Test the nodes attributes number consistency over all internal data structures of the pytorch Data object
        that involve nodes.
        """

        self.assertEqual(self.ppi_pytorch_graph.x.size(dim=1), self.ppi_pytorch_graph.node_feature_labels.shape[0],
                         "graph.x.size(dim=1) and graph.node_feature_labels.shape[0] must be equal")

    def test_edges_nr(self):
        self.assertEqual(sum((1, 2, 3)), 6, "Should be 6")

    def tearDown(self):
        pass


def test_suite():
    """
    Define the test suite - the sequence of the tests that will be executed
    """

    suite_test = unittest.TestSuite()

    suite_test.addTest(DataConsistency("test_nodes_nr"))
    suite_test.addTest(DataConsistency("test_node_attr_nr"))

    runner = HTMLTestRunner(
        log=True,
        verbosity=2,
        output=os.path.join("tests", "test_reports"),
        title='Test report',
        report_name='report',
        open_in_browser=True,
        description="HTMLTestReport")
    runner.run(suite_test)


########################################################################################################################
# MAIN =================================================================================================================
########################################################################################################################
if __name__ == '__main__':
    test_suite()
