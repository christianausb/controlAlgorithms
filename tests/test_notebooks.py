
# example_test.py
from testbook import testbook


def verify_test_results(test_results):
    for r in test_results:
        if not r[1]:
            raise BaseException('Test ' + r[0] + ' failed')

@testbook('examples/sysident.ipynb', execute=True)
def test_sysident(tb):
   assert True

@testbook('examples/state_est_pendulum.ipynb', execute=True)
def test_state_est_pendulum(tb):
   assert True

@testbook('examples/trajectory_optim_integrator.ipynb', execute=True)
def test_trajectory_optim(tb):
#   verify_test_results(tb.test_results)
   assert True

@testbook('examples/trajectory_optim_cart_pendulum.ipynb', execute=True)
def test_trajectory_optim(tb):
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
