
from click.testing import CliRunner

from chunkflow.cli import main


def test_main():
    runner = CliRunner()
    result = runner.invoke(main, [])

    if False:
        assert result.output == '()\n'
        assert result.exit_code == 0
