
# example_test.py
from testbook import testbook


@testbook('examples/sysident.ipynb', execute=True)
def test_sysident(tb):
   assert True

@testbook('examples/state_est_pendulum.ipynb', execute=True)
def test_state_est_pendulum(tb):
   assert True

@testbook('examples/trajectory_optim.ipynb', execute=True)
def test_trajectory_optim(tb):
   assert True

@testbook('examples/trajectory_optim_flow.ipynb', execute=True)
def test_trajectory_optim_flow(tb):
   assert True

@testbook('examples/trajectory_optim_pendulum.ipynb', execute=True)
def test_trajectory_optim_pendulum(tb):
   assert True
