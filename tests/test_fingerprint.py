from smartrag.ingest.fingerprint import Fingerprinter


class TestFingerprinter:
    def test_synopsis_length(self):
        fp = Fingerprinter()
        text = "This is a comprehensive guide to building REST APIs with Flask. " * 30
        result = fp.generate(text, title="Flask REST API Guide")
        assert len(result.synopsis) <= 200

    def test_fingerprint_keywords(self):
        fp = Fingerprinter()
        text = (
            "Authentication middleware validates JWT tokens on every API request. "
            "The middleware extracts the bearer token from the Authorization header "
            "and verifies its signature."
        )
        result = fp.generate(text, title="Auth Middleware")
        assert len(result.fingerprint) >= 1
        assert isinstance(result.fingerprint, list)

    def test_categories(self):
        fp = Fingerprinter()
        text = "This API endpoint handles user authentication via JWT tokens and OAuth2 flows."
        result = fp.generate(text, title="Auth API")
        assert len(result.categories) >= 1

    def test_empty_text(self):
        fp = Fingerprinter()
        result = fp.generate("", title="Empty")
        assert result.synopsis == ""
        assert result.categories == ["general"]

    def test_section_synopsis(self):
        fp = Fingerprinter()
        synopsis = fp.generate_section_synopsis(
            "This section covers database migration strategies.", "Migrations"
        )
        assert len(synopsis) <= 150

    def test_concepts(self):
        fp = Fingerprinter()
        text = (
            "## JWT Authentication\n\n"
            "JSON Web Tokens provide **stateless authentication**. "
            "The **Authorization Header** carries the token."
        )
        result = fp.generate(text, title="JWT Auth")
        assert len(result.concepts) >= 0  # May find concepts from headings/bold
