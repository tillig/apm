"""Unit tests for --update auth pre-flight probe in pipeline.py (#1015)."""

import subprocess  # noqa: F401
from unittest.mock import MagicMock, patch

import pytest

from apm_cli.install.errors import AuthenticationError


def _make_dep(host="dev.azure.com", repo_url="myorg/myproject/_git/myrepo"):
    dep = MagicMock()
    dep.host = host
    dep.repo_url = repo_url
    dep.port = None
    dep.is_azure_devops.return_value = True
    dep.explicit_scheme = None
    dep.is_insecure = False
    dep.ado_organization = "myorg"
    dep.ado_project = "myproject"
    dep.ado_repo = "myrepo"
    return dep


def _make_ctx(update_refs=True, deps=None):
    ctx = MagicMock()
    ctx.deps_to_install = deps or [_make_dep()]
    ctx.update_refs = update_refs
    return ctx


def _make_resolver(auth_scheme="basic", token="pat", git_env=None):  # noqa: S107
    resolver = MagicMock()
    dep_ctx = MagicMock()
    dep_ctx.token = token
    dep_ctx.auth_scheme = auth_scheme
    dep_ctx.git_env = git_env or {}
    resolver.resolve_for_dep.return_value = dep_ctx
    resolver.build_error_context.return_value = "    Diagnostic payload"
    return resolver


class TestUpdatePreflightRejectsBadAuth:
    """Pre-flight raises AuthenticationError when git ls-remote returns 401."""

    @patch("subprocess.run")
    def test_auth_failure_raises(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=128,
            stderr="fatal: Authentication failed (401)",
            stdout="",
        )
        from apm_cli.install.pipeline import _preflight_auth_check

        ctx = _make_ctx()
        resolver = _make_resolver()

        with pytest.raises(AuthenticationError) as exc_info:
            _preflight_auth_check(ctx, resolver, verbose=False)

        assert "No files were modified" in exc_info.value.diagnostic_context
        assert "apm.yml" in exc_info.value.diagnostic_context

    @patch("subprocess.run")
    def test_auth_failure_message_mentions_host(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=128,
            stderr="fatal: unable to access (403)",
            stdout="",
        )
        from apm_cli.install.pipeline import _preflight_auth_check

        ctx = _make_ctx()
        resolver = _make_resolver()

        with pytest.raises(AuthenticationError) as exc_info:
            _preflight_auth_check(ctx, resolver, verbose=False)

        # Bounded full-phrase assertion (see CodeQL note in test_validation_ado_bearer.py).
        assert str(exc_info.value) == "Authentication failed for dev.azure.com"


class TestUpdatePreflightPassesGoodAuth:
    """Pre-flight succeeds when git ls-remote returns rc=0."""

    @patch("subprocess.run")
    def test_good_auth_no_exception(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stderr="",
            stdout="abc123\trefs/heads/main\n",
        )
        from apm_cli.install.pipeline import _preflight_auth_check

        ctx = _make_ctx()
        resolver = _make_resolver()

        # Should not raise
        _preflight_auth_check(ctx, resolver, verbose=False)


class TestPreflightSkippedForGitHubDeps:
    """github.com deps are skipped (they use the API probe with unauth fallback)."""

    @patch("subprocess.run")
    def test_github_deps_skipped(self, mock_run):
        dep = _make_dep(host="github.com", repo_url="owner/repo")
        ctx = _make_ctx(deps=[dep])
        resolver = _make_resolver()

        from apm_cli.install.pipeline import _preflight_auth_check

        _preflight_auth_check(ctx, resolver, verbose=False)
        mock_run.assert_not_called()


class TestPreflightClustersDeduplicate:
    """Multiple deps on the same (host, org) only probe once."""

    @patch("subprocess.run")
    def test_deduplication(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stderr="", stdout="")

        dep1 = _make_dep(host="dev.azure.com", repo_url="myorg/projA/_git/repoA")
        dep2 = _make_dep(host="dev.azure.com", repo_url="myorg/projB/_git/repoB")
        ctx = _make_ctx(deps=[dep1, dep2])
        resolver = _make_resolver()

        from apm_cli.install.pipeline import _preflight_auth_check

        _preflight_auth_check(ctx, resolver, verbose=False)
        assert mock_run.call_count == 1


def _make_generic_dep(host="gitlab.internal.corp", repo_url="org/repo"):
    """Create a mock dep for a generic (non-GitHub, non-ADO) host."""
    dep = MagicMock()
    dep.host = host
    dep.repo_url = repo_url
    dep.port = None
    dep.is_azure_devops.return_value = False
    dep.explicit_scheme = None
    dep.is_insecure = False
    return dep


class TestPreflightGenericHostAllowsCredentialHelpers:
    """Generic hosts (GHES, GitLab, etc.) must not block credential helpers (#1082)."""

    @patch("subprocess.run")
    def test_generic_host_env_omits_git_config_global(self, mock_run):
        """The probe env for generic hosts must NOT set GIT_CONFIG_GLOBAL.

        GIT_CONFIG_GLOBAL=/dev/null blocks credential helpers configured in
        ~/.gitconfig, which is the primary auth mechanism for non-GitHub hosts.
        """
        mock_run.return_value = MagicMock(returncode=0, stderr="", stdout="")

        dep = _make_generic_dep(host="ghes.corp.example.com")
        ctx = _make_ctx(deps=[dep])
        resolver = _make_resolver(token="some-token")

        from apm_cli.install.pipeline import _preflight_auth_check

        _preflight_auth_check(ctx, resolver, verbose=False)

        assert mock_run.call_count == 1
        call_env = mock_run.call_args[1]["env"]
        # GIT_CONFIG_GLOBAL must not be set (or must not point to /dev/null)
        # so that credential helpers from ~/.gitconfig can function.
        git_config_global = call_env.get("GIT_CONFIG_GLOBAL")
        assert git_config_global is None or git_config_global != "/dev/null"

    @patch("subprocess.run")
    def test_generic_host_env_allows_askpass(self, mock_run):
        """GIT_ASKPASS must not be 'echo' for generic hosts.

        Setting GIT_ASKPASS=echo prevents credential helpers from being
        invoked by git.
        """
        mock_run.return_value = MagicMock(returncode=0, stderr="", stdout="")

        dep = _make_generic_dep(host="gitlab.example.com")
        ctx = _make_ctx(deps=[dep])
        resolver = _make_resolver(token="some-token")

        from apm_cli.install.pipeline import _preflight_auth_check

        _preflight_auth_check(ctx, resolver, verbose=False)

        call_env = mock_run.call_args[1]["env"]
        assert call_env.get("GIT_ASKPASS") != "echo"

    @patch("subprocess.run")
    def test_generic_host_auth_failure_still_raises(self, mock_run):
        """Auth failures on generic hosts still raise AuthenticationError."""
        mock_run.return_value = MagicMock(
            returncode=128,
            stderr="fatal: Authentication failed for 'https://ghes.corp.example.com/'",
            stdout="",
        )

        dep = _make_generic_dep(host="ghes.corp.example.com")
        ctx = _make_ctx(deps=[dep])
        resolver = _make_resolver(token="some-token")

        from apm_cli.install.pipeline import _preflight_auth_check

        with pytest.raises(AuthenticationError) as exc_info:
            _preflight_auth_check(ctx, resolver, verbose=False)

        assert "ghes.corp.example.com" in str(exc_info.value)

    @patch("subprocess.run")
    @patch("apm_cli.deps.github_downloader.GitHubPackageDownloader._build_noninteractive_git_env")
    def test_ado_host_does_not_use_noninteractive_env(self, mock_ni_env, mock_run):
        """ADO hosts should NOT use the noninteractive env (they use token in URL)."""
        mock_run.return_value = MagicMock(returncode=0, stderr="", stdout="")
        mock_ni_env.return_value = {"MARKER": "noninteractive"}

        dep = _make_dep(host="dev.azure.com", repo_url="myorg/myproject/_git/myrepo")
        ctx = _make_ctx(deps=[dep])
        resolver = _make_resolver(token="ado-pat")

        from apm_cli.install.pipeline import _preflight_auth_check

        _preflight_auth_check(ctx, resolver, verbose=False)

        # _build_noninteractive_git_env should NOT be called for ADO hosts
        mock_ni_env.assert_not_called()
